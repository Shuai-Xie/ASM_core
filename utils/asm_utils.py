import torch
from torchvision.transforms import functional as F
from PIL import Image
from tqdm import tqdm
import numpy as np
from utils.box_utils import box_iou


@torch.no_grad()
def infer(model, img_path, device):
    img = Image.open(img_path).convert('RGB')
    img_tensor = F.to_tensor(img)
    detection = model([img_tensor.to(device)])[0]

    # parse detection, label-1 与 gt 对应
    boxes, labels, scores = detection['boxes'], detection['labels'] - 1, detection['scores']
    return boxes.cpu(), labels.cpu(), scores.cpu()  # 转到 cpu, 为了使用 torch.where()


def keep_idxs_by(condition):
    keep_idxs = torch.where(condition, torch.tensor(1), torch.tensor(0)).nonzero().reshape(-1)  # list 类型
    return keep_idxs


def rejudge_certain_anns(boxes, labels, cer_idxs, uncer_idxs, iou_thre=0.5):
    """
    boxes, labels: detection 全部输出结果
    iou(certain_box, uncertain_box) 再比较其 label
        如果 label 不同，说明 model 对该位置不确定 (top n 不确定)
        如果 label 相同，保留 cer box 即可
    iou_thre 判断出的 box, 执行 nms，把相对于 cer_box 的 uncertain boxes 去掉
    iou_thre 控制着 uncertain 阈值，越低更多 certain 样本化为 uncertain，更多样本流入 AL
    @return: 筛选过的 cer_boxes, cer_labels
    """
    keep_idxs = []
    cer_boxes, cer_labels = boxes[cer_idxs], labels[cer_idxs]
    uncer_boxes, uncer_labels = boxes[uncer_idxs], labels[uncer_idxs]

    for cer_idx, cer_box in enumerate(cer_boxes):
        certain = True
        # boxes, torch.float32
        cer_box_rep = cer_box.repeat((uncer_boxes.size()[0], 1))  # repeat 便于向量比较
        ious = box_iou(cer_box_rep, uncer_boxes)

        # >= iou_thre 的 uncertain_box 下标
        mask = torch.where(ious >= iou_thre, torch.tensor(1), torch.tensor(0)).nonzero()
        if mask.size()[0] > 0:
            # uncertain 中有 label 不同的，cer
            if cer_labels[cer_idx] not in uncer_labels[mask]:
                certain = False

        if certain:
            keep_idxs.append(cer_idx)

        # 相当于 nms，将上面 mask 的 box 去掉，不用于下轮判断，加快执行
        unmask = [idx for idx in range(uncer_boxes.size()[0]) if idx not in mask]
        if len(unmask) == 0:
            # 已经没有 uncertain boxes 进行对比了，直接保存剩下所有的 cer_box
            keep_idxs += [idx for idx in range(cer_idx + 1, len(cer_boxes))]  # 从 cer_idx+1 到最后
            break

        uncer_boxes = uncer_boxes[unmask]
        uncer_labels = uncer_labels[unmask]

    return cer_idxs[keep_idxs]  # 最后保留的 box


def update_sa_anns(boxes, labels, cer_idxs, gt_anns, iou_thre):
    """
    比较 certain_anns 和 GT_anns 的 iou，定下哪些 anns 需要用 cer_anns 替换
    """
    cer_boxes, cer_labels = boxes[cer_idxs], labels[cer_idxs]
    gt_boxes, gt_labels = gt_anns
    sa_boxes, sa_labels = gt_boxes.clone(), gt_labels  # 保存 gt，在此基础上更新

    cer_keep_idxs = []
    gt_sl_idxs = []  # 记录 gt 中哪些 ann 是 model ann 的

    for cer_idx, cer_box in enumerate(cer_boxes):
        cer_box_rep = cer_box.repeat((gt_boxes.size()[0], 1))
        ious = box_iou(cer_box_rep, gt_boxes)
        val, gt_idx = torch.topk(ious, k=1)
        # 与 gt 阈值 >= 0.8，并且 label 相同
        # 找到后，gt_boxes 不动态删了，要保证 gt_idx 的对应
        if cer_labels[cer_idx] == gt_labels[gt_idx] and val >= iou_thre:
            sa_boxes[gt_idx] = cer_box
            cer_keep_idxs.append(cer_idx)  # 保存再次筛选后的 idx
            gt_sl_idxs.append(gt_idx)

    # 与 gt ann 比较后，再次更新 certain_idxs，相当于 human judge 结果
    cer_idxs = cer_idxs[cer_keep_idxs]

    return cer_idxs, gt_sl_idxs, sa_boxes, sa_labels


@torch.no_grad()
def detect_unlabel_imgs(model, batch_unlabel_anns, device,
                        certain_thre, uncertain_thre,
                        judge_iou_thre=0.5, gt_iou_thre=0.8):
    """
    todo: 并行化; CONF_THRESH 动态调节; 样本 uncertain 评级每次选出最优的一组？
    """
    model.eval()
    batch_sa_anns = []
    batch_gt_sl_idxs = []  # sl idxs in gt anns
    batch_sl_ratio, batch_al_ratio = [], []
    for ann in tqdm(batch_unlabel_anns):  # idx as unlabel image id
        # model 直接 out 的 boxes
        # [postprocess: box_score_thresh=0.05, box_nms_thresh=0.5]
        boxes, labels, scores = infer(model, ann['filepath'], device)  # labels is int64

        # 分开 certain / uncertain boxes, idxs 1D tensor
        keep_idxs = keep_idxs_by(scores >= uncertain_thre)  # conf 分段
        certain_idxs = keep_idxs_by(scores >= certain_thre)
        uncertain_idxs = torch.tensor([idx for idx in keep_idxs if idx not in certain_idxs],
                                      dtype=torch.int64)

        if uncertain_idxs.size()[0] > 0:
            # 1.与 uncertain box 比较后，更新 certain_idxs, 根据 iou(cer_box, uncer_box)
            certain_idxs = rejudge_certain_anns(boxes, labels,
                                                certain_idxs, uncertain_idxs, iou_thre=judge_iou_thre)

        # 如果不存在 uncertain_idxs，直接与 gt anns 比较
        # 2.与 gt ann 比较后，再次更新 certain_idxs，相当于 human judge 结果
        gt_anns = torch.tensor(ann['boxes'], dtype=torch.float32), torch.tensor(ann['labels'])
        # sa_anns = gt_anns + cer_anns，即部分 GT 用 certain anns 替换
        # len(certain_idxs) = len(gt_sl_idxs)
        certain_idxs, gt_sl_idxs, sa_boxes, sa_labels = update_sa_anns(boxes, labels,
                                                                       certain_idxs, gt_anns, iou_thre=gt_iou_thre)
        # 添加 sa_ann
        ann['boxes'] = sa_boxes.numpy()  # 转 numpy
        ann['labels'] = sa_labels.numpy()
        batch_sa_anns.append(ann)

        # 记录 单张图像 SL/AL anns 占比
        sl_ratio = len(gt_sl_idxs) / len(sa_labels)
        batch_sl_ratio.append(sl_ratio)
        batch_al_ratio.append(1 - sl_ratio)
        batch_gt_sl_idxs.append(gt_sl_idxs)

    return batch_sa_anns, batch_sl_ratio, batch_al_ratio, batch_gt_sl_idxs
