from torchvision.datasets import VOCDetection, VOCSegmentation
from datasets.transforms import get_transform
from datasets.voc.configs import class2names, names2class
from torchvision.transforms import functional as F
from utils.plt_utils import plt_bbox


def parse_boxes_labels(objs, n2c):
    boxes, labels = [], []
    if not isinstance(objs, list):  # 字典也是 Iterable
        objs = [objs]
    for obj in objs:
        box = obj['bndbox']
        box = [box['xmin'], box['ymin'], box['xmax'], box['ymax']]
        box = list(map(int, box))
        boxes.append(box)
        labels.append(n2c[obj['name']])
    return boxes, labels


def demo_VOCDetection():
    # 这种 dataset 一旦建立就不太好截断，动态更新下次训练的传入数据集了
    # samples read from txt, 如果使用这个数据集，需要动态存储要使用的图像到 txt
    dataset = VOCDetection(
        root='/nfs/xs/Datasets/VOC2007',
        year='2007',
        image_set='trainval',
        download=False,  # already download
        transforms=get_transform(train=True)  # transforms: 自己传入
    )
    # print(len(voc_det))  # trainval: 5011

    for idx, (img, target) in enumerate(dataset):
        img = F.to_pil_image(img)
        objects = target['annotation']['object']
        print(objects)
        boxes, labels = parse_boxes_labels(objects, names2class)
        scores = [1] * len(boxes)
        plt_bbox(img, boxes, labels, scores, class2names)

        if idx > 4:
            break


def demo_VOCSegmentation():
    pass


demo_VOCSegmentation()
