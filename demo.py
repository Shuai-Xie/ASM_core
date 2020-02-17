import torch
from torchvision.transforms import functional as F
from net.faster_rcnn import get_model
from utils.plt_utils import plt_bbox
from datasets.configs import class2names
import numpy as np
from PIL import Image
import os
import argparse


def load_detection_model(ckpt):
    model = get_model(backbone=args.backbone,
                      input_size=(args.input_h, args.input_w),
                      num_classes=args.num_classes,
                      self_pretrained=True)
    model.load_state_dict(torch.load(ckpt))
    print('load pretrain:', ckpt)
    model = model.to(device)
    return model


def get_hook(name, hooks):
    hooks[name] = {}

    def hook(module, input, output):
        # 只要 module 的输出
        hooks[name]['output'] = output

    return hook


def to_numpy(tensor):
    return tensor.cpu().numpy()


@torch.no_grad()  # 与 tensor.detach() 都能不计算 grad
def infer(model, img, score_thre=0.7):
    img_tensor = F.to_tensor(img)  # cv, PIL 都可
    # model 内部进行了后处理，就算用 hook 拿到 cls_logtis 也得不到输出的
    detection = model([img_tensor.to(device)])[0]  # only 1 img
    # roi_heads.postprocess_detections() 除去了一些 box
    # box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100

    # todo 怎么判断生成的 box 是否全了? 结合 Dataturks 虽然只要把 high conf 标注上?
    # 从半自动标注角度，只要模型生成的 box 在 Dataturks 上展示即可
    # 从模型的训练角度，结合 Dataturks 后，无论 SL 还是 AL 的 box 都需要 user 看一遍?
    # 能否细化到 SL 中不确定项，给 user 更多提示? 需要 rank 这些 box，才能给出 最应该给 user 标注的图像
    # 先把简易版本的，与 Dataturks 联动，给师弟
    # 为不确定的 box 插入 Dataturks 时加上 note?
    # 控制批次，阶段性引入：为 model 带来性能提升，提升 SL 性能，并减少单次 AL 进入时间
    # early stop, ap_shift_thre 小一点？

    # parse detection
    boxes, labels, scores = detection['boxes'], detection['labels'], detection['scores']

    # scores 保留了所有 box 的 prob 相对于所有类的softmax最大值，但没有相对于各个类的得分
    # 所以如果1个 box 的出现 前2类 prob 都很高，就筛选不出来了；根据 conf 中间和 box_iou 决断
    print(scores.shape, scores)
    keep_idxs = np.where(scores > score_thre)[0]

    return boxes[keep_idxs], labels[keep_idxs], scores[keep_idxs]


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
    model = load_detection_model(ckpt='output/res50_al/res50_epoch_5.pth')
    model.eval()

    demo_dir = '/nfs/xs/Datasets/VOC2007/VOCdevkit/VOC2007/JPEGImages'
    img_list = os.listdir(demo_dir)
    random.shuffle(img_list)
    for img in img_list[:20]:
        print(img)
        img_path = os.path.join(demo_dir, img)
        img = Image.open(img_path)  # 不用 cvt RGB, to_tensor 会做
        boxes, labels, scores = infer(model, img, score_thre=0)  # =0 保留模型所有输出的 box 看看，这样能留下 low conf?
        plt_bbox(img, boxes, labels, scores, class2names)
