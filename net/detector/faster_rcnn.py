import torchvision
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor, fasterrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator

"""
MaskRCNN -> FasterRCNN -> GeneralizedRCNN (forward in here)

forward():
    backbone -> rpn -> roi_heads -> postprocess(recover box)
    # rpn use RPN params
    # roi_heads use Box and Mask params

GeneralizedRCNN has forward(), son class hasn't
if train:
    input: images (list[Tensor]), target (list[Dict[Tensor]])
    return: losses (dict[Tensor]) {'proposal_losses', 'detector_losses'}
if test:
    input: images (list[Tensor]), target = None
    return: detections (list[BoxList])

FasterRCNN(GeneralizedRCNN) -> fasterrcnn_resnet50_fpn
    # RPN, Box
if train:
    input: images, target_dict {'boxes', 'labels'}
    return: losses (dict[Tensor]) {'proposal_losses', 'detector_losses'}
if test:
    input: images (list[Tensor]), target = None
    return: detections (list[BoxList])

MaskRCNN(FasterRCNN) -> maskrcnn_resnet50_fpn
    # RPN, Box, Mask
if train:
    input: images, target_dict { 'boxes', 'labels', 'masks' }
    return: losses (dict[Tensor]) {'proposal_losses', 'detector_losses'}
"""


def faster_rcnn_mobile(input_size, num_classes, self_pretrained):
    if not self_pretrained:
        print('load mobilenet_v2 backbone pretrained on ImageNet')
    backbone = torchvision.models.mobilenet_v2(pretrained=not self_pretrained).features
    backbone.out_channels = 1280
    # anchors
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

    anchor_generator = AnchorGenerator(sizes=((32,), (64,), (128,), (256,), (512,),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
    # need this, as mobilenet out 1 level features
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],  # can be multi
                                                    output_size=7,
                                                    sampling_ratio=2)
    # model will do normalize and resize itself
    # box_nms_thresh used during inference
    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       min_size=input_size[0], max_size=input_size[1],
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    return model


def faster_rcnn_res50(input_size, num_classes, self_pretrained):
    """
    torchvision 自带的 fasterrcnn_resnet50_fpn 在 coco 上训练的，只提供了 resnet50
        函数内部 resnet_fpn_backbone() 设置 backbone 类型
        定义 layer2,3,4 之前层 require_grad=False, 之后 BackboneWithFPN() 添加了 FPN
    """
    if not self_pretrained:
        print('load res50 backbone pretrained on COCO')

    model = fasterrcnn_resnet50_fpn(
        pretrained=not self_pretrained,
        min_size=input_size[0], max_size=input_size[1],
        # rpn_anchor_generator=anchor_generator,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 更新 `fasterrcnn_resnet50_fpn` 内 FastRCNNPredictor, num_classes 自定义
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features,
                                                      num_classes=num_classes)
    return model


def get_model(backbone, input_size, num_classes, self_pretrained=False):
    if backbone == 'res50':
        return faster_rcnn_res50(input_size, num_classes, self_pretrained)
    elif backbone == 'mobile':
        return faster_rcnn_mobile(input_size, num_classes, self_pretrained)
    else:
        raise ValueError('not implement such backbone!')


def see_anchors():
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    # copy 5 times, 相当于每个 anchor_size 都有这些 ratio
    print(aspect_ratios)
    num_anchors_per_location = [len(s) * len(a) for s, a in zip(anchor_sizes, aspect_ratios)]
    print(num_anchors_per_location)  # [3, 3, 3, 3, 3]
    num_anchors = num_anchors_per_location[0]  # 3


if __name__ == '__main__':
    pass
