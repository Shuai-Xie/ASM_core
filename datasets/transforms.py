from torchvision.transforms import functional as F
from PIL import ImageEnhance
import random

"""
faster rcnn 自己会 scale, normalize
"""


def get_transform(train=True):
    if train:
        transforms = [RandomEnhance(), RandomHorizontalFlip(0.5), ToTensor()]
    else:
        transforms = [ToTensor()]
    return Compose(transforms)


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose:
    """
    as sample (img, target), not use torchvision.transforms.Compose
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):  # pillow img
        for t in self.transforms:  # a bunch of transforms
            img, target = t(img, target)
        return img, target


class RandomEnhance:
    def __call__(self, img, target):
        """
        F.adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation
        """
        if random.random() < 0.5:
            img = ImageEnhance.Brightness(img).enhance(0.5 + random.random())
        if random.random() < 0.5:
            img = ImageEnhance.Color(img).enhance(0.5 + random.random())
        if random.random() < 0.5:
            img = ImageEnhance.Contrast(img).enhance(0.5 + random.random())
        return img, target


class RandomHorizontalFlip:
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, img, target):
        if random.random() < self.prob:
            height, width = img.size
            img = F.hflip(img)
            # VOC 默认数据集的得处理下
            if target is not None:
                if "boxes" in target:
                    bbox = target["boxes"]
                    bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # 0,2 - x0, x1
                    target["boxes"] = bbox
                if "masks" in target:
                    target["masks"] = target["masks"].flip(-1)
                if "keypoints" in target:
                    keypoints = target["keypoints"]
                    keypoints = _flip_coco_person_keypoints(keypoints, width)
                    target["keypoints"] = keypoints
        return img, target


class ToTensor:
    def __call__(self, img, target):
        img = F.to_tensor(img)
        return img, target
