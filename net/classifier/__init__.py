from .densenet import DenseNet121, DenseNet161, DenseNet169, DenseNet201
from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152


def get_model(net, num_classes):
    return globals()[net](num_classes)
