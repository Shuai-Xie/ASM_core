import torch
from torchvision.transforms import functional as F
from net.faster_rcnn import get_model
from utils.plt_utils import plt_bbox
from datasets.configs import class2names
import numpy as np
from PIL import Image
import os
import argparse
from utils.model_utils import load_model
from datasets.voc_parser import load_data
from utils.asm_utils import detect_unlabel_imgs
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='demo')
    parser.add_argument('--gpus', default='1', type=str)
    # model
    parser.add_argument('--backbone', default='res50', type=str, help='res50, mobile')
    parser.add_argument('--num_classes', default=21, type=int)
    parser.add_argument('--input_h', default=480, type=int)
    parser.add_argument('--input_w', default=640, type=int)
    args = parser.parse_args()

    # gpu
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda:{}'.format(args.gpus)) if torch.cuda.is_available() else torch.device('cpu')

    # model
    model = get_model(backbone=args.backbone,
                      input_size=(args.input_h, args.input_w),
                      num_classes=args.num_classes,
                      self_pretrained=True)
    # model = load_model(model, ckpt_path='output/res50_al/res50_epoch_5.pth')
    model.load_state_dict(torch.load('output/res50_al/res50_epoch_5.pth'))
    model = model.to(device)
    model.eval()

    eval_anns = load_data('VOC2007', split='test')  # 4592
    random.seed(1)  # 控制复现
    random.shuffle(eval_anns)
    eval_anns = eval_anns[:20]

    # demo_dir = '/nfs/xs/Datasets/VOC2007/VOCdevkit/VOC2007/JPEGImages'
    # img_list = os.listdir(demo_dir)

    for ann in eval_anns:
        img = Image.open(ann['filepath'])

        # GT boxes
        boxes, labels = ann['boxes'], ann['labels']
        scores = [1] * len(boxes)
        plt_bbox(img, boxes, labels, scores, class2names)

        # human & model anns
        batch_sa_anns, batch_sl_num, batch_al_num, batch_gt_sl_idxs = detect_unlabel_imgs(model, [ann], device,
                                                                                          certain_thre=0.8,
                                                                                          uncertain_thre=0.3)
        # SA boxes
        sa_ann, sl_idxs = batch_sa_anns[0], batch_gt_sl_idxs[0]
        boxes, labels = sa_ann['boxes'], sa_ann['labels']
        scores = [1] * len(boxes)
        for idx in sl_idxs:
            scores[idx] = 0.8  # SL 标识，并不是真正 prob

        plt_bbox(img, boxes, labels, scores, class2names)
