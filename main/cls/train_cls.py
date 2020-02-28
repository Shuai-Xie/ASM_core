import os
import sys

# 加入项目根目录，后面自定义 pkg 就可正常 import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.cifar.configs import fetch_ori_data
from datasets.cls_datasets import CIFAR

from net.classifier import get_model

from main.cls.engine import train_one_epoch, evaluate

from utils.model_utils import load_model, save_model
from utils.process_utils import lasso_shift
from utils.time_utils import get_curtime


def train_model(dataloader, start_epoch, acc_range, acc_shift_thre):
    for idx, epoch in enumerate(range(start_epoch, args.num_epoches)):
        # train
        train_acc = train_one_epoch(model, dataloader, device, epoch,
                                    writer, optimizer, criterion)
        # eval after each train
        eval_acc = evaluate(model, dataloader_eval, device)
        writer.add_scalars('Train/accuracy', {
            'train': train_acc,
            'eval': eval_acc,
        }, global_step=epoch)

        # record eval states
        acc_records['acc'].append(eval_acc)
        if len(acc_records['acc']) >= acc_range:
            acc_shift = lasso_shift(acc_records['acc'][-acc_range:])
        else:
            acc_shift = 0
        acc_records['acc_shift'].append(acc_shift)
        writer.add_scalar('Train/acc_shift', acc_shift, global_step=epoch)
        print('epoch: {}, train_acc: {}, eval_acc: {}, acc_shift: {}'.format(
            epoch, train_acc, eval_acc, acc_shift))

        if 0 < acc_shift < acc_shift_thre:  # eval acc 稳定时结束
            best_epoch, best_acc = epoch, eval_acc
            ckpt_path = os.path.join(model_save_dir, '{}_epoch_{}.pth'.format(args.net, epoch))
            save_model(ckpt_path, model, best_epoch, best_acc, optimizer)
            print('best epoch: {}, accuracy: {}'.format(best_epoch, best_acc))
            return epoch, best_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASM Classification on cifar10')
    parser.add_argument('--gpus', default='0', type=str)
    # model
    parser.add_argument('--net', default='ResNet18', type=str,
                        help='available models in net/classifier dir')
    parser.add_argument('--num_classes', default=10, type=int)
    # train
    parser.add_argument('--num_epoches', default=50, type=int, help='total epoches')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--check_step', default=1, type=int, help='epoch steps to save model')
    parser.add_argument('--ckpt', type=str, help='pretrained model')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    # early stop
    parser.add_argument('--acc_range', default=2, type=int,
                        help='acc_range to get lasso acc_shfit')
    parser.add_argument('--acc_shift_thre', default=0.0005, type=float,
                        help='acc_shift threshold to stop asm model training')

    params = [
        '--gpus', '0',
        '--net', 'ResNet18',
        '--batch_size', '128',
        # '--ckpt', 'output/res50_sa/res50_epoch_5.pth',
    ]
    args = parser.parse_args(params)

    # gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda:{}'.format(args.gpus)) if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.empty_cache()
    cudnn.benchmark = True

    # prepare dataset
    cifar10 = fetch_ori_data('cifar10', root='/nfs/xs/Datasets/CIFAR10')
    classes, class_to_idx = cifar10['classes'], cifar10['class_to_idx']
    x_train, y_train = cifar10['train']  # build label/unlabel set, 50000
    x_test, y_test = cifar10['test']  # build valid set, 10000

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),  # 水平翻转
        transforms.ToTensor(),  # normalize after tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataset_train = CIFAR(x_train, y_train, transform=transform_train)
    dataloader_train = DataLoader(dataset_train, args.batch_size, shuffle=True, num_workers=4)
    dataset_eval = CIFAR(x_test, y_test, transform=transform_test)
    dataloader_eval = DataLoader(dataset_eval, batch_size=100, shuffle=False, num_workers=4)
    print('load initial_trainset and evalset done!')

    # model
    model = get_model(args.net, args.num_classes)
    model = model.to(device)
    # optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # loss
    criterion = nn.CrossEntropyLoss()

    # tensorbaord & model_save
    writer = SummaryWriter(log_dir='runs/cls_{}_{}'.format(args.net, get_curtime()))
    acc_records = {'acc': [], 'acc_shift': []}
    model_save_dir = os.path.join('output', 'cls_{}_{}'.format(args.net, get_curtime()))
    os.makedirs(model_save_dir, exist_ok=True)

    start_epoch = 0
    if args.ckpt:
        print('load pretrain model')
        model, optimizer, best_epoch, best_acc = load_model(model, args.ckpt, optimizer)
        start_epoch = best_epoch + 1

    train_model(dataloader_train, start_epoch, args.acc_range, args.acc_shift_thre)
