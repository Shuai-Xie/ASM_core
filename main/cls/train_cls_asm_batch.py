"""
CEAL: https://github.com/dhaalves/CEAL_keras/blob/master/CEAL_keras.py
- detect all unlabel pool, not batch
"""
import os
import sys

# 加入项目根目录，后面自定义 pkg 就可正常 import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.cifar.configs import fetch_ori_data
from datasets.cls_datasets import CIFAR

from net.classifier import get_model

from main.cls.select_criteria import get_select_criterion_fn, get_high_confidence_samples
from main.cls.engine import train_one_epoch, evaluate, detect_unlabel_imgs, Delta_Scheduler

from utils.model_utils import load_model, save_model
from utils.process_utils import lasso_shift
from utils.time_utils import get_curtime


def empty_x():
    return np.empty([0] + list(x_initial.shape[1:]), dtype='uint8')  # (0,32,32,3)


def empty_y():
    return np.empty([0], dtype='int64')


def update_asm_dataloader(writer, epoch):
    """
    两次引入新样本： AL: uncer_idxs + SL: hc_idxs
    """
    global DU, DL  # pool, init_label

    print('classify unlabel imgs in DU')
    y_pred_porb = detect_unlabel_imgs(model, DU[0], DU[1], device)  # (N,10)
    # expand DL with uncertain samples from DU
    _, uncer_idxs = select_fn(y_pred_porb, args.uncertain_samples_size)  # 1000

    # only get uncertain
    batch_DU = np.take(DU[0], uncer_idxs, axis=0), np.take(DU[1], uncer_idxs, axis=0)
    batch_DH = empty_x(), empty_y()

    if args.cost_effective:  # 使用 high confidence
        # 内部调用 entropy() 也是 sorted
        hc_idxs, hc_labels = get_high_confidence_samples(y_pred_porb, delta_scheduler.delta)
        writer.add_scalar('ASM/entropy_delta', delta_scheduler.delta, global_step=epoch)

        # remove samples also selected through uncertain, zip 1个 for 同时遍历
        hc = np.array([[i, l] for i, l in zip(hc_idxs, hc_labels) if i not in uncer_idxs])
        if len(hc) > 0:
            # topK from hc
            topK = min(len(hc), args.uncertain_samples_size)
            cer_idxs = hc[:, 0][:topK]
            batch_DH = np.take(DU[0], cer_idxs, axis=0), hc[:, 1][:topK]  # hc_imgs, hc_labels
            hc_gt_labels = np.take(DU[1], cer_idxs, axis=0)
            # 只计算取出 batch 的 acc
            hc_accuracy = len(np.where(batch_DH[1] == hc_gt_labels)[0]) / len(hc_gt_labels)
            writer.add_scalar('ASM/hc_samples', len(hc), global_step=epoch)
            writer.add_scalar('ASM/hc_accuracy', hc_accuracy, global_step=epoch)
            # todo: clean topK from DU?
            # DU = np.delete(DU[0], cer_idxs, axis=0), np.delete(DU[1], cer_idxs, axis=0)

    random_idxs = np.random.permutation(range(len(DL[0])))[:args.uncertain_samples_size]
    batch_DL = np.take(DL[0], random_idxs, axis=0), np.take(DL[1], random_idxs, axis=0)

    # concat new trainset: label + uncertain + high conf
    x_train, y_train = np.concatenate((batch_DL[0], batch_DU[0], batch_DH[0])), \
                       np.concatenate((batch_DL[1], batch_DU[1], batch_DH[1]))
    writer.add_scalar('ASM/total_samples', len(y_train), global_step=epoch)
    # new dataloader
    dataset = CIFAR(x_train, y_train, transform_train)
    dataloader = DataLoader(dataset, args.batch_size, shuffle=True, num_workers=4)
    print('create new trainset')

    # 更新完 dataloader
    # uncertain samples 加入 DL，并从 DU 删除
    DL = np.append(DL[0], np.take(DU[0], uncer_idxs, axis=0), axis=0), \
         np.append(DL[1], np.take(DU[1], uncer_idxs, axis=0), axis=0)
    DU = np.delete(DU[0], uncer_idxs, axis=0), np.delete(DU[1], uncer_idxs, axis=0)
    # 并没有删除检测的 certain_idxs

    # entropy threshold decay
    # train_model() 设置 finetune_interval, 每几个 epoch 更新 dataloader 和其他参数
    delta_scheduler.step()  # updae delta

    return dataloader


def train_model(dataloader, start_epoch, acc_range, acc_shift_thre, asm=False):
    for idx, epoch in enumerate(range(start_epoch, args.num_epoches)):
        if asm and idx % args.finetune_interval == 0:
            dataloader = update_asm_dataloader(writer, epoch)
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
        writer.add_scalar('Train/eval_acc_shift', acc_shift, global_step=epoch)
        print('epoch: {}, train_acc: {}, eval_acc: {}, acc_shift: {}'.format(
            epoch, train_acc, eval_acc, acc_shift))

        if not asm:
            if eval_acc > 0.5:
                return epoch, eval_acc
        else:
            if 0 < acc_shift < acc_shift_thre:  # eval acc 稳定时结束
                best_epoch, best_acc = epoch, eval_acc
                ckpt_path = os.path.join(model_save_dir, '{}_ckpt.pth'.format(args.net))  # overwrite?
                save_model(ckpt_path, model, best_epoch, best_acc, optimizer)
                print('best epoch: {}, accuracy: {}'.format(best_epoch, best_acc))
                return epoch, best_acc

    return -1, 0


def init_dataset(init_ratio):
    # prepare dataset
    # x: np,(N,32,32,3); y: list [N]
    cifar10 = fetch_ori_data('cifar10', root='/nfs/xs/Datasets/CIFAR10')
    classes, class_to_idx = cifar10['classes'], cifar10['class_to_idx']
    x_train, y_train = cifar10['train']  # build label/unlabel set, 50000
    x_test, y_test = cifar10['test']  # build valid set, 10000

    # 根据 initial_annotated_perc 划分 initial(label)/pool(unlabel)
    split = int(len(x_train) * init_ratio)
    x_initial, y_initial = x_train[:split], y_train[:split]  # 截断划分，数量基本均衡
    x_pool, y_pool = x_train[split:], y_train[split:]

    return x_pool, y_pool, x_initial, y_initial, x_test, y_test, classes, class_to_idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASM Classification on cifar10')
    parser.add_argument('--gpus', default='0', type=str)
    # model
    parser.add_argument('--net', default='ResNet18', type=str,
                        help='available models in net/classifier dir')
    parser.add_argument('--num_classes', default=10, type=int)
    # train
    parser.add_argument('--num_epoches', default=50, type=int, help='max total epoches')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--check_step', default=1, type=int, help='epoch steps to save model')
    parser.add_argument('--ckpt', type=str, help='pretrained model')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    # early stop
    parser.add_argument('--acc_range', default=2, type=int,
                        help='acc_range to get lasso acc_shfit')
    parser.add_argument('--acc_shift_pre', default=0.001, type=float,
                        help='acc_shift threshold to stop pretrain model training')
    parser.add_argument('--acc_shift_asm', default=0.0001, type=float,
                        help='acc_shift threshold to stop asm model training')
    # asm / uncertain
    parser.add_argument('-i', '--initial_annotated_ratio', default=0.2, type=float,  # 5000 samples
                        help='Initial Annotated Samples Ratio. default: 0.1')
    parser.add_argument('-K', '--uncertain_samples_size', default=2000, type=int,
                        help='Uncertain samples selection size. default: 1000')
    parser.add_argument('-uc', '--uncertain_criterion', default='ms', type=str,
                        help='Uncertain selection Criteria:\n'
                             'rs(Random Sampling)\n'
                             'lc(Least Confidence)\n'
                             'ms(Margin Sampling)\n'
                             'en(Entropy)')
    # ceal high confidence
    parser.add_argument('-ce', '--cost_effective', default=True,
                        help="whether to use Cost Effective high confidence sample pseudo-labeling. default: True")
    parser.add_argument('-delta_begin', default=0.05, type=float,  # prob > 0.99
                        help="High confidence samples selection threshold. default: 0.05")
    parser.add_argument('-delta_end', default=0.05, type=float,  # prob > 0.99
                        help="High confidence samples selection threshold. default: 0.05")
    parser.add_argument('-t', '--finetune_interval', default=1, type=int,
                        help="Fine-tuning interval. default: 1")
    params = [
        '--gpus', '0',
        '--net', 'ResNet18',
        '--batch_size', '128',
        '-t', '2'
        # '--ckpt', 'output/res50_sa/res50_epoch_5.pth',
    ]
    args = parser.parse_args(params)

    # gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda:{}'.format(args.gpus)) if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.empty_cache()
    cudnn.benchmark = True

    # prepare dataset
    print('split label(initial) & unlabel(pool) dataset')
    x_pool, y_pool, x_initial, y_initial, x_test, y_test, classes, class_to_idx = \
        init_dataset(init_ratio=args.initial_annotated_ratio)
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
    dataset_train = CIFAR(x_initial, y_initial, transform=transform_train)
    dataloader_train = DataLoader(dataset_train, args.batch_size, shuffle=True, num_workers=4)
    dataset_eval = CIFAR(x_test, y_test, transform=transform_test)
    dataloader_eval = DataLoader(dataset_eval, batch_size=100, shuffle=False, num_workers=4)
    print('load initial_trainset and evalset done!')

    # symbols in paper
    DU = x_pool, y_pool  # unlabel pool
    DL = x_initial, y_initial  # label initial
    DH = empty_x(), empty_y()  # empty to save high confidence samples

    # model
    model = get_model(args.net, args.num_classes)
    model = model.to(device)
    # optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # loss
    criterion = nn.CrossEntropyLoss()
    # uncertain select fn
    select_fn = get_select_criterion_fn(args.uncertain_criterion)

    # tensorbaord & model_save
    writer = SummaryWriter(log_dir='runs/cls_{}_{}_{}'.format(args.uncertain_criterion, args.net, get_curtime()))
    acc_records = {'acc': [], 'acc_shift': []}
    model_save_dir = os.path.join('output', 'cls_{}_{}_{}'.format(args.uncertain_criterion, args.net, get_curtime()))
    os.makedirs(model_save_dir, exist_ok=True)

    if args.ckpt:
        print('load pretrain model')
        model, optimizer, best_epoch, best_acc = load_model(model, args.ckpt, optimizer)
    else:
        print('train model on initial dataset')
        best_epoch, best_acc = train_model(dataloader_train, 0, args.acc_range, args.acc_shift_pre, asm=False)

    # delta to control hc samples
    delta_scheduler = Delta_Scheduler(args.delta_begin, args.delta_end,
                                      max_steps=(args.num_epoches - best_epoch) // args.finetune_interval)
    # begin asm
    start_epoch = best_epoch + 1
    train_model(dataloader_train, start_epoch, args.acc_range, args.acc_shift_asm, asm=True)
