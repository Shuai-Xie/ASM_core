import math
import sys
import time
import torch
import torchvision.models.detection.mask_rcnn
from tools.coco_utils import get_coco_api_from_dataset
from tools.coco_eval import CocoEvaluator
from tools import utils

target_keys = {
    'detect': ['boxes', 'labels'],
    'segment': ['boxes', 'labels', 'masks']
}


def train_one_epoch(model, optimizer, dataloader, device, epoch, print_freq,
                    writer=None, begin_step=0):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000  # 很小的 warm，从 0.001 逐步增加到 0.004，增加函数自定义
        warmup_iters = min(1000, len(dataloader) - 1)  # warm 迭代次数
        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    cnt = 0
    for images, targets in metric_logger.log_every(dataloader, print_freq, header):
        # input: images(list[Tensor]), target(list[Dict[Tensor]])
        images = [image.to(device) for image in images]
        targets = [
            {k: v.to(device) for k, v in t.items()}
            for t in targets
        ]

        # losses(dict[Tensor])  {'proposal_losses', 'detector_losses'}
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum([loss for loss in loss_dict_reduced.values()])

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        loss_value = losses_reduced.item()
        if cnt % print_freq == 0:  # 可以 每 10 iter 记录一下 loss
            writer.add_scalar('Train/loss', losses_reduced.item(), global_step=begin_step + cnt)
        cnt += 1

        if not math.isfinite(loss_value):
            print("Loss is {}, stop training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        if lr_scheduler is not None:  # epoch=0时，warmup_lr_scheduler
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, dataloader, device, writer=None, epoch=0):
    model.eval()

    n_threads = torch.get_num_threads()  # 8 线程, model on gpu, eval on cpu
    torch.set_num_threads(1)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(dataloader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    # eval can be mulit process too.
    for image, targets in metric_logger.log_every(dataloader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [
            {k: v.to(device) for k, v in t.items()}
            for t in targets
        ]
        torch.cuda.synchronize()  # 多线程同步?
        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, outputs)
        }
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    # coco_evaluator.summarize()  # 'IoU metric:' print here!
    eval_stats = coco_evaluator.summarize()

    writer.add_scalar('Accuracy/AP_50', eval_stats[1], global_step=epoch)
    writer.add_scalar('Accuracy/AP_75', eval_stats[2], global_step=epoch)

    torch.set_num_threads(n_threads)

    # return coco_evaluator
    return {
        'ap_50': eval_stats[1],
        'ap_75': eval_stats[2]
    }
