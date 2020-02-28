from torchvision.datasets import cifar
from utils.plt_utils import plt_samples
import numpy as np

"""
cifar: https://www.cs.toronto.edu/~kriz/cifar.html

CIFAR10
- split: train(5000) + test(1000)
- The dataset is divided into 5 training batches and 1 test batch, each with 10000 images.

CIFAR100
- split: train(500) + test(100)
- grouped into 20 superclasses.

cifar.CIFAR10/CIFAR100 加载数据时
先下载 tar.gz 上传，第1次 verified 花时间，后面就加载很快了
内部解析 data_batch_1, data_batch_2... 文件, pickle.load(f)
"""


def fetch_ori_data(dataset, root):
    """
    fetch original data from torchvision CIFAR10/100 class
    @return:
        data: np.ndarray, (50000, 32, 32, 3), all data is here!
        targets: list, 50000, all label is here!
        classes: list, [name1, name2, ...]
        class_to_idx: dict, {name:idx, ...}
    """
    if dataset == 'cifar10':
        trainset = cifar.CIFAR10(root, train=True, download=True)
        testset = cifar.CIFAR10(root, train=False, download=True)
    elif dataset == 'cifar100':
        trainset = cifar.CIFAR100(root, train=True, download=True)
        testset = cifar.CIFAR100(root, train=False, download=True)
    else:
        raise ValueError('no this dataset')

    cifar_ = {
        'train': (trainset.data, np.array(trainset.targets)),
        'test': (testset.data, np.array(testset.targets)),
        'classes': trainset.classes,
        'class_to_idx': trainset.class_to_idx
    }
    return cifar_


def show_samples(inputs, targets, class2names):
    plt_samples(inputs, targets, class2names)


if __name__ == '__main__':
    cifar10 = fetch_ori_data('cifar10', root='/nfs/xs/Datasets/CIFAR10')
    x_train, y_train = cifar10['train']
    print(x_train.dtype)  # uint8
    print(y_train.dtype)  # int64
