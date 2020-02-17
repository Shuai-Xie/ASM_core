from torchvision.transforms import functional as F
import torch
import numpy as np
from PIL import Image, ImageEnhance

"""Todo
其实没必要，faster_rcnn 模型自己会做 normalize 
"""


def get_transform(train=True):
    if train:
        transforms = [RandomEnhance(), ToTensor()]
    else:
        transforms = [ToTensor()]
    return Compose(transforms)


class Compose:
    """
    as sample (image, target), not use torchvision.transforms.Compose
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:  # a bunch of transforms
            image, target = t(image, target)
        return image, target


class RandomEnhance:
    def __call__(self, img, target):
        if np.random.random() < 0.5:
            img = ImageEnhance.Brightness(img).enhance(0.5 + np.random.random())
        if np.random.random() < 0.5:
            img = ImageEnhance.Color(img).enhance(0.5 + np.random.random())
        if np.random.random() < 0.5:
            img = ImageEnhance.Contrast(img).enhance(0.5 + np.random.random())
        return img, target


class RandomHorizontalFlip:
    def __call__(self, img, target):
        if np.random.random() < 0.5:
            img, target = horizontal_flip(img, target)
        return img, target


class ToTensor:
    def __call__(self, img, target):
        img = F.to_tensor(img)  # use torchvision func to_tensor
        return img, target
