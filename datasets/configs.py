class2names = [  # 0-19
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
names2class = {
    name: idx for idx, name in enumerate(class2names)
}

voc_img_num = {
    'VOC2007_trainval': 5011,
    'VOC2007_test': 4592,
    'VOC2012_trainval': 11540
}

# instances num
VOC2007_trainval = {
    'aeroplane': 331,
    'bicycle': 418,
    'bird': 599,
    'boat': 398,
    'bottle': 634,
    'bus': 272,
    'car': 1644,
    'cat': 389,
    'chair': 1432,
    'cow': 356,
    'diningtable': 310,
    'dog': 538,
    'horse': 406,
    'motorbike': 390,
    'person': 5447,
    'pottedplant': 625,
    'sheep': 353,
    'sofa': 425,
    'train': 328,
    'tvmonitor': 367
}

VOC2007_test = {
    'aeroplane': 311,
    'bicycle': 389,
    'bird': 576,
    'boat': 393,
    'bottle': 657,
    'bus': 254,
    'car': 1541,
    'cat': 370,
    'chair': 1374,
    'cow': 329,
    'diningtable': 299,
    'dog': 530,
    'horse': 395,
    'motorbike': 369,
    'person': 5227,
    'pottedplant': 592,
    'sheep': 311,
    'sofa': 396,
    'train': 302,
    'tvmonitor': 361
}

VOC2012_trainval = {
    'aeroplane': 954,
    'bicycle': 790,
    'bird': 1221,
    'boat': 999,
    'bottle': 1482,
    'bus': 637,
    'car': 2364,
    'cat': 1227,
    'chair': 2906,
    'cow': 702,
    'diningtable': 747,
    'dog': 1541,
    'horse': 750,
    'motorbike': 751,
    'person': 10129,
    'pottedplant': 1099,
    'sheep': 994,
    'sofa': 786,
    'train': 656,
    'tvmonitor': 826
}
