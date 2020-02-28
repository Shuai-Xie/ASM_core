import os
import torch
from torch.utils.tensorboard import SummaryWriter

from datasets.VOC import VOC_Dataset
from datasets.transforms import get_transform
from datasets.voc_parser import load_data

from net.faster_rcnn import get_model
from tools.utils import collate_fn
from tools.engine import train_one_epoch, evaluate
from utils.process_utils import lasso_shift
from utils.io_utils import dump_json
from utils.time_utils import get_curtime
from utils.model_utils import load_model, save_model

import argparse

"""
train model on VOC2007, get pretrain model
then begin active learning on VOC2012
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train FasterRCNN on VOC2007')
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
    parser.add_argument('--ap_shift_thre', default=0.02, type=float, help='ap_shift threshold to stop training')

    params = [
        '--gpus', '1',
        '--backbone', 'res50',
        '--num_classes', '21',  # 20+bg
        '--batch_size', '4',
        '--check_step', '1',
        '--ckpt', 'output_opt/res50/res50_epoch_1.pth',
    ]
    args = parser.parse_args(params)

    # gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda:{}'.format(args.gpus)) if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    # get model
    self_pretrained = False if args.ckpt is None else True
    model = get_model(backbone=args.backbone,
                      input_size=(args.input_h, args.input_w),
                      num_classes=args.num_classes,
                      self_pretrained=self_pretrained)
    model = model.to(device)

    # model save
    model_save_dir = os.path.join('output_opt', args.backbone)
    os.makedirs(model_save_dir, exist_ok=True)

    # optimizer
    params = [p for p in model.parameters() if p.requires_grad]  # conv,layer1 不更新
    optimizer = torch.optim.SGD(params, lr=0.004, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)  # 1/10
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoches)  # 1/10
    start_epoch = 0

    if self_pretrained:
        model, optimizer, start_epoch = load_model(model, args.ckpt, optimizer)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1, last_epoch=start_epoch - 1)

    # prepare dataset
    train_anns = load_data('VOC2007', split='trainval')  # 5011
    train_anns = train_anns[:20]
    eval_anns = load_data('VOC2007', split='test')  # 4592
    eval_anns = eval_anns[:2]

    dataset_train = VOC_Dataset(data=train_anns, split=None, transforms=get_transform(True))
    dataset_eval = VOC_Dataset(data=eval_anns, split=None, transforms=get_transform(False))
    dataloader = torch.utils.data.DataLoader(dataset_train,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             # use True if gpu is enough, Tensor转义到GPU显存加快
                                             # False: 4748MiB, True: 5208MiB
                                             drop_last=True,
                                             num_workers=4,
                                             collate_fn=collate_fn)
    dataloader_eval = torch.utils.data.DataLoader(dataset_eval,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=4,
                                                  collate_fn=collate_fn)
    print('load dataset done!')

    writer = SummaryWriter(log_dir='runs/{}_pretrain'.format(get_curtime()))

    ap_records = {
        'ap_50': [],
        'ap_75': [],
        'ap_shift': []
    }

    for epoch in range(start_epoch, start_epoch + args.num_epoches):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, dataloader, device, epoch, print_freq=2,
                        writer=writer, begin_step=(epoch - 1) * len(dataloader))
        # store & update lr
        writer.add_scalar('Train/lr', optimizer.param_groups[0]["lr"], global_step=epoch)
        lr_scheduler.step()

        # evals
        evals = evaluate(model, dataloader_eval, device, writer, epoch)

        # states
        ap_records['ap_50'].append(evals['ap_50'])
        ap_records['ap_75'].append(evals['ap_75'])
        ap_shift = max(0, lasso_shift(ap_records[args.ap][-args.ap_range:]))
        ap_records['ap_shift'].append(ap_shift)

        writer.add_scalar('Accuracy/AP_shift', ap_shift, global_step=epoch)

        if epoch % args.check_step == 0:
            ckpt_path = os.path.join(model_save_dir, '{}_epoch_{}.pth'.format(args.backbone, epoch))
            save_model(ckpt_path, model, epoch, optimizer)

            if epoch >= 1:
                break
