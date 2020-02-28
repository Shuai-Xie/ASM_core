import torch
from torchvision.models.resnet import (ResNet, BasicBlock, Bottleneck,
                                       resnet18, resnet50, resnext50_32x4d)


def load_pretrain_features(model, features):
    model.conv1 = features.conv1
    model.bn1 = features.bn1
    model.relu = features.relu
    model.maxpool = features.maxpool
    model.layer1 = features.layer1
    model.layer2 = features.layer2
    model.layer3 = features.layer3
    model.layer4 = features.layer4
    return model


def _resnet18(num_classes, pretrained=True):
    # default no pretrain weights model, original initialize
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    if pretrained:
        features = resnet18(pretrained)
        model = load_pretrain_features(model, features)
    return model


def _resnet50(num_classes, pretrained=True):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    if pretrained:
        features = resnet50(pretrained)
        model = load_pretrain_features(model, features)
    return model


def _resnext50(num_classes, pretrained=True):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes,
                   groups=32, width_per_group=8)  # width = int(planes * (base_width / 64.)) * groups
    if pretrained:
        features = resnext50_32x4d(pretrained)
        model = load_pretrain_features(model, features)
    return model


def get_model(backbone, num_classes, pretrained=False):
    # 不用指定 input size， fc 会拉成 1D vector, 逐步得到 num_classes logits
    # 不能完全用 默认 model，在 load pretrain 时，需要处理 fc 层不同问题
    if backbone == 'resnet18':
        return _resnet18(num_classes, pretrained)
    elif backbone == 'resnet50':
        return _resnet50(num_classes, pretrained)
    elif backbone == 'resnext50':
        return _resnext50(num_classes, pretrained)
    else:
        raise ValueError('not implement such backbone!')


if __name__ == '__main__':
    img = torch.rand(1, 3, 224, 224)
    model = get_model('resnet18', num_classes=10, pretrained=True)
    res = model(img)
    print(res.size())
