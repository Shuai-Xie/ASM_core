import os
import torch
from torch.utils.tensorboard import SummaryWriter

from datasets.VOC import VOC_Dataset
from datasets.transforms import get_transform
from datasets.voc_parser import load_data

from net.faster_rcnn import get_model
from tools.utils import collate_fn
from tools.engine import train_one_epoch, evaluate
from utils.asm_utils import detect_unlabel_imgs
from utils.process_utils import lasso_shift
from utils.io_utils import dump_json
import argparse

import numpy as np
import random
from utils.time_utils import get_curtime

"""
有 pretrain 的 model
"""


def get_learning_tag():
    tag = ''
    if args.enable_al and args.enable_sl:
        print('use al and sl samples')
        tag = 'as'
    elif args.enable_al:
        print('only use al samples')
        tag = 'al'
    elif args.enable_sl:
        print('only using sl samples is not meaningful!')
        tag = 'sl'
    return tag


def get_anns_per_img(ann_list):
    # 获得 train_anns 中每张 img 平均的 ann 数目
    return np.mean([len(ann['labels']) for ann in ann_list])


def update_asm_dataloader(tag, device, writer, epoch):
    """
    update asm_train_anns and dataloader
    @param tag: learning tag: as, al, sl
    @param device: model infer device
    @param writer: SummaryWriter
    @param epoch: current epoch idx
    @return: data_loader with sl/al results
    """
    # detect on unlabel voc2012
    # target: sl_idxs increase, al_idxs reduce!
    print('detect on unlable data')

    global label_anns, unlabel_anns
    sl_anns, al_anns = detect_unlabel_imgs(model, unlabel_anns, device, args.conf_thre)
    # update label anns, label + unlabel 之和不变
    label_anns += al_anns  # 模拟加入人工标注数据
    unlabel_anns = sl_anns  # 除去 al_anns 的 sl_anns 模型自标注数据再作为 unlabel

    # tensorboard SL/AL 样本变化趋势
    writer.add_scalar('Samples/SL', len(sl_anns), global_step=epoch)
    writer.add_scalar('Samples/AL', len(al_anns), global_step=epoch)
    writer.add_scalars('Samples/Total', {
        'label': len(label_anns),
        'unlabel': len(unlabel_anns),
    }, global_step=epoch)

    asm_train_anns = label_anns + sl_anns if tag == 'as' else label_anns
    dataset = VOC_Dataset(data=asm_train_anns, split=None, transforms=get_transform(True))
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=4,
                                             collate_fn=collate_fn)

    print('epoch {}: update train dataset'.format(epoch))
    return dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple ASM with FasterRCNN on VOC2012')
    parser.add_argument('--gpus', default='0', type=str)
    # model
    parser.add_argument('--backbone', default='res50', type=str, help='res50, mobile')
    parser.add_argument('--num_classes', type=int)
    parser.add_argument('--input_h', default=480, type=int)
    parser.add_argument('--input_w', default=640, type=int)
    # train
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_epoches', default=10, type=int, help='total epoches to asm')
    parser.add_argument('--check_step', default=2, type=int, help='epoch steps to save model')
    parser.add_argument('--ckpt', type=str, help='pretrained model on VOC2007')
    # early stop
    parser.add_argument('--ap', default='ap_50', type=str, help='ap_50, ap_75')
    parser.add_argument('--ap_thre', default=0.5, type=float, help='ap threshold to save ckpt')
    parser.add_argument('--ap_range', default=3, type=int, help='ap range to get lasso ap_shfit')
    parser.add_argument('--ap_shift_thre', default=0.0001, type=float, help='ap_shift threshold to stop training')
    # asm
    parser.add_argument('--conf_thre', default=0.7, type=float, help='threshold to keep sl samples')  # keep prob
    parser.add_argument('--enable_al', default=True, type=bool, help='whether to use al process')
    parser.add_argument('--enable_sl', default=False, type=bool, help='whether to use sl process')

    params = [
        '--gpus', '1',
        '--backbone', 'res50',
        '--num_classes', '21',  # 20+bg
        '--batch_size', '4',
        '--check_step', '1',
        '--ckpt', 'output/res50/res50_epoch_6.pth',
        '--conf_thre', '0.9',  # todo: keep prob, 实际应用调整
    ]
    args = parser.parse_args(params)

    # gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda:{}'.format(args.gpus)) if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    # load pretrain model on voc2007, then do active learning on voc2012
    # asm 必须先有 pretrain model
    model = get_model(backbone=args.backbone,
                      input_size=(args.input_h, args.input_w),
                      num_classes=args.num_classes,
                      self_pretrained=True)
    model.load_state_dict(torch.load(args.ckpt))
    print('load pretrain:', args.ckpt)
    # model = torch.nn.DataParallel(model).to(device)  # data parallel
    model = model.to(device)

    # update data loader, allow al and sl data
    learning_tag = get_learning_tag()  # ['as', 'al', 'sl']

    # optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    # 减小初始学习率 0.004 -> 0.0004
    optimizer = torch.optim.SGD(params, lr=0.00004, momentum=0.9, weight_decay=0.0005)
    # lr scheduler which decreases the learning rate by 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # model save
    model_save_dir = os.path.join('output', '{}_{}_smooth'.format(args.backbone, learning_tag))
    os.makedirs(model_save_dir, exist_ok=True)
    ap_out_path = os.path.join(model_save_dir, 'ap_records_{}.json'.format(args.backbone))
    # vis writer
    writer = SummaryWriter(log_dir='runs/{}_{}_smooth'.format(get_curtime(), learning_tag))

    # prepare dataset
    # view VOC2012 trainval as unlabel dataset
    voc2007_anns = load_data('VOC2007', split='trainval')  # 5011
    voc2012_anns = load_data('VOC2012', split='trainval')  # 11540
    label_anns = voc2007_anns[:1000]  # same to train.py / less already labeled data
    unlabel_anns = voc2012_anns[:1000]  # unlabel
    # asm_eval_anns
    asm_eval_anns = load_data('VOC2007', split='test')  # 4592
    asm_eval_anns = asm_eval_anns[:300]

    dataset_eval = VOC_Dataset(data=asm_eval_anns, split=None, transforms=get_transform(False))
    dataloader_eval = torch.utils.data.DataLoader(dataset_eval,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=4,
                                                  collate_fn=collate_fn)
    print('load eval dataset done!')

    # eval stats 0: pretrain model
    evaluate(model, dataloader_eval, device, writer, epoch=0)

    # 根据 model infer results 逐步引入 al_anns, 更新 label/unlabel anns
    dataloader = update_asm_dataloader(learning_tag, device, writer, epoch=0)

    ap_records = {
        'ap_50': [],
        'ap_75': [],
        'ap_shift': []
    }

    # begin train again!
    for epoch in range(1, args.num_epoches + 1):
        # train for one epoch, print every 2 batchs
        train_one_epoch(model, optimizer, dataloader, device, epoch, print_freq=2,
                        writer=writer, begin_step=(epoch - 1) * len(dataloader))
        # update the learning rate
        lr_scheduler.step()

        # evals
        evals = evaluate(model, dataloader_eval, device, writer, epoch)

        # states
        ap_50, ap_75 = evals['ap_50'], evals['ap_75']
        ap_records['ap_50'].append(ap_50)
        ap_records['ap_75'].append(ap_75)
        ap_shift = max(0, lasso_shift(ap_records[args.ap][-args.ap_range:]))
        ap_records['ap_shift'].append(ap_shift)

        writer.add_scalar('Accuracy/AP_shift', ap_shift, global_step=epoch)

        if epoch % args.check_step == 0 and evals[args.ap] > args.ap_thre:
            torch.save(model.state_dict(), os.path.join(
                model_save_dir, '{}_epoch_{}.pth'.format(args.backbone, epoch)))

            # update training data afer each eval
            dataloader = update_asm_dataloader(learning_tag, device, writer, epoch)

            # finetune AP 变化范围很小
            if 0 < ap_shift < args.ap_shift_thre:  # break and save ap records
                dump_json(ap_records, ap_out_path)
                break
