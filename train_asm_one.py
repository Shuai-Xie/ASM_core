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
from utils.time_utils import get_curtime
from utils.model_utils import load_model, save_model

import argparse
import random
import numpy as np


def get_learning_tag():
    tag = ''
    if args.enable_al and args.enable_sl:
        print('use sl and al samples')
        tag = 'sa'
    elif args.enable_al:
        print('only use al samples')
        tag = 'al'
    elif args.enable_sl:
        print('only using sl samples is not meaningful!')
        tag = 'sl'
    return tag


def update_asm_dataloader(tag, batch_idx, writer, epoch):
    """
    update asm_train_anns and dataloader
        可传入 eval acc 作为更新 CONF_THRESH 的依据
    @param tag: learning tag: as, al, sl
    @param batch_idx: 增量样本的批次 idx, 每次 K 个
    @param writer: SummaryWriter
    @param epoch: current epoch idx
    @return: data_loader with sl/al results
    """
    # detect on unlabel voc2012
    # target: sl_idxs increase, al_idxs reduce!
    print('detect on unlable data')

    global label_anns, unlabel_anns, sa_anns  # 作为 global 变量，时刻更新

    # 根据 batch_idx 分批次引入 unlabel samples
    # K 控制每个 epoch 引入的新样本数量
    # todo: 判断每次引入的 sample 学习完毕? 设置 AP 阈值，逐渐增加
    #       引入新 anns 后，eval_anns 也要引入新的测试数据，将 batch_sa_anns 选出部分加入?
    upper = min((batch_idx + 1) * args.K, len(unlabel_anns))
    batch_unlabel_anns = unlabel_anns[batch_idx * args.K:upper]

    # SL/AL anns
    batch_sa_anns, batch_sl_ratio, batch_al_ratio, _ = detect_unlabel_imgs(model, batch_unlabel_anns, device,
                                                                           args.certain_thre, args.uncertain_thre)
    # todo: 可以选择 AL ratio 高的样本?
    #       将 sa_anns 保存起来 下一轮 batch_detect 引入 ratio 高的?
    #       不设 ratio thre，而选择 top 100 默认就带入了 ratio 逐渐降低的过程
    # 从 init anns 中也选取 K 个与 sa_anns 共同组成 new trainset
    random.seed()  # 每次随机不一样
    random_label_anns = random.sample(label_anns, args.K)
    asm_train_anns = batch_sa_anns + random_label_anns
    # 更新 label_anns，下一 batch 也会筛选一些 init anns 之外的样本
    label_anns += batch_sa_anns

    # tensorboard SL/AL 样本 ratio 变化趋势
    writer.add_scalars('ASM/sample_ratio', {
        'SL': np.mean(batch_sl_ratio),
        'AL': np.mean(batch_al_ratio),
    }, global_step=epoch)

    dataset = VOC_Dataset(data=asm_train_anns, split=None, transforms=get_transform(True))
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=4,
                                             collate_fn=collate_fn)

    print('epoch {}: update train dataset'.format(epoch))
    return dataloader


def train_model(dataloader, start_epoch, ap_shift_thre, asm=True):
    batch_samples_idx = 0  # 从 unlabel_anns 切出 K 个样本
    if asm:
        dataloader = update_asm_dataloader(learning_tag, batch_samples_idx, writer, epoch=start_epoch)

    for epoch in range(start_epoch, args.num_epoches):  # epoch 要从0开始，内部有 warm_up
        # train
        train_one_epoch(model, optimizer, dataloader, device, epoch, print_freq=10,
                        writer=writer, begin_step=epoch * len(dataloader))
        # store & update lr
        writer.add_scalar('ASM/lr', optimizer.param_groups[0]["lr"], global_step=epoch)
        lr_scheduler.step()
        # eval after each train
        evals = evaluate(model, dataloader_eval, device, writer, epoch)

        # states
        ap_records['ap_50'].append(evals['ap_50'])
        ap_records['ap_75'].append(evals['ap_75'])
        if len(ap_records[args.ap]) >= args.ap_range:
            ap_shift = lasso_shift(ap_records[args.ap][-args.ap_range:])
        else:
            ap_shift = 0
        ap_records['ap_shift'].append(ap_shift)

        writer.add_scalar('Accuracy/AP_shift', ap_shift, global_step=epoch)

        if evals[args.ap] > args.ap_thre:
            ckpt_path = os.path.join(model_save_dir, '{}_epoch_{}.pth'.format(args.backbone, epoch))
            save_model(ckpt_path, model, epoch, optimizer)

            if asm:
                batch_samples_idx += 1
                dataloader = update_asm_dataloader(learning_tag, batch_samples_idx, writer, epoch)

            if 0 < ap_shift < ap_shift_thre:  # break and save ap records
                best_idx_in_range = ap_records[args.ap].index(max(ap_records[args.ap][-args.ap_range:]))
                best_epoch = epoch - args.ap_range + 1 + best_idx_in_range
                # 从 ap_range 中选取最优 epoch
                ap_records['ap_50'] = ap_records['ap_50'][:best_epoch + 1]
                ap_records['ap_75'] = ap_records['ap_75'][:best_epoch + 1]
                ap_records['ap_shift'] = ap_records['ap_shift'][:best_epoch + 1]
                return best_epoch


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
    parser.add_argument('--num_epoches', default=20, type=int, help='max total epoches')
    parser.add_argument('--check_step', default=2, type=int, help='epoch steps to save model')
    parser.add_argument('--ckpt', type=str, help='pretrained model on VOC2007')
    # early stop
    parser.add_argument('--ap', default='ap_50', type=str, help='ap_50, ap_75')
    parser.add_argument('--ap_thre', default=0.5, type=float, help='ap threshold to save ckpt')
    parser.add_argument('--ap_range', default=3, type=int, help='ap range to get lasso ap_shfit')
    parser.add_argument('--ap_shift_pretrain', default=0.05, type=float, help='ap_shift threshold to stop training')
    parser.add_argument('--ap_shift_asm', default=0.001, type=float, help='ap_shift threshold to stop training')
    # asm
    parser.add_argument('--certain_thre', default=0.8, type=float, help='threshold to keep sl samples')  # keep prob
    parser.add_argument('--uncertain_thre', default=0.3, type=float, help='threshold to keep al samples')  # [0.3]
    parser.add_argument('--K', default=400, type=int, help='add new unlabel samples each iter')
    parser.add_argument('--enable_al', default=True, type=bool, help='whether to use al process')
    parser.add_argument('--enable_sl', default=True, type=bool, help='whether to use sl process')

    params = [
        '--gpus', '0',
        '--backbone', 'res50',
        '--num_classes', '21',  # 20+bg
        '--batch_size', '8',
        '--check_step', '1',  # eval step
        # '--ckpt', 'output/res50_sa/res50_epoch_5.pth',
    ]
    args = parser.parse_args(params)

    # tag
    learning_tag = get_learning_tag()  # ['sa', 'al', 'sl']

    # gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda:{}'.format(args.gpus)) if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.empty_cache()

    # model
    self_pretrained = False if args.ckpt is None else True
    model = get_model(backbone=args.backbone,
                      input_size=(args.input_h, args.input_w),
                      num_classes=args.num_classes,
                      self_pretrained=self_pretrained)
    model = model.to(device)

    # optimizer, 和 model 一样必须先定义好，之后可从 ckpt load 参数
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # prepare dataset
    # voc2007 分成 label, unlabel
    voc2007_anns = load_data('VOC2007', split='trainval')  # 5011
    label_anns = voc2007_anns[:1000]  # initial train
    unlabel_anns = voc2007_anns[1000:]  # incremently AL 分批实现
    sa_anns = []
    eval_anns = load_data('VOC2007', split='test')  # 4592
    random.seed(1)  # 每次随机一样，增加复现性
    eval_anns = random.sample(eval_anns, 1000)
    # dataloader
    dataset_train = VOC_Dataset(data=label_anns, split=None, transforms=get_transform(True))
    dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=4,
                                                   collate_fn=collate_fn)
    dataset_eval = VOC_Dataset(data=eval_anns, split=None, transforms=get_transform(False))
    dataloader_eval = torch.utils.data.DataLoader(dataset_eval,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=4,
                                                  collate_fn=collate_fn)
    print('load dataset done!')

    # 检测参数
    writer = SummaryWriter(log_dir='runs/{}'.format(get_curtime()))
    ap_records = {'ap_50': [], 'ap_75': [], 'ap_shift': []}

    # model save
    model_save_dir = os.path.join('output', '{}_{}'.format(args.backbone, get_curtime()))
    os.makedirs(model_save_dir, exist_ok=True)

    if self_pretrained:
        print('load pretrain model')
        model, optimizer, start_epoch = load_model(model, args.ckpt, optimizer)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5, last_epoch=start_epoch - 1)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoches)
    else:
        print('train model on initial dataset')
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)  # 默认 -1
        # todo: 引入新 anns 应该带来 学习率的改变?
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoches)
        best_epoch = train_model(dataloader_train, 0, args.ap_shift_pretrain, asm=False)  # 内部使用 lr_scheduler
        start_epoch = best_epoch + 1

    print('asm begin epoch:', start_epoch)
    train_model(dataloader_train, start_epoch, args.ap_shift_asm, asm=True)
