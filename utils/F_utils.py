from PIL import Image
import matplotlib.pyplot as plt
from datasets.transforms import get_transform
from torchvision.transforms import functional as F
import torchvision.transforms as transforms
import os

"""
https://pytorch.org/docs/stable/torchvision/transforms.html#functional-transforms
"""


def show_img(img):
    plt.imshow(img)
    plt.show()


def test_hflip(img):
    show_img(img)
    img = F.hflip(img)  # 左右翻转
    show_img(img)


def test_transform():
    trans = get_transform(train=True)

    demo_dir = '../data/demo'
    for img in os.listdir(demo_dir):
        img = Image.open(os.path.join(demo_dir, img))
        show_img(img)
        img, _ = trans(img, target=None)
        img = F.to_pil_image(img)
        show_img(img)


if __name__ == '__main__':
    test_transform()
    pass
