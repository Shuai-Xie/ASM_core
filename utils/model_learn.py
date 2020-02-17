import torch
from collections import OrderedDict
from net.faster_rcnn import get_model


def get_hook(name, hooks):
    """
    @param name: 作为 hooks 的 dict key 使用
    @param hooks: {} 全局变量，可在多个 register_forward_hook 使用同一个 hooks
    """

    hooks[name] = {}

    def hook(module, input, output):
        """
        一般用于 model forward() 获取 中间结果, 如果想反向传播可去掉 detach() 只有 tensor 可用
            module: 不用赋值，对应层 register_forward_hook 时会设置
        """
        # module 层输入
        if isinstance(input, tuple) or isinstance(input, list):  # 某层返回多个结果
            hooks[name]['input'] = (inp for inp in input)
        elif isinstance(input, OrderedDict):
            hooks[name]['input'] = OrderedDict({k: v for k, v in input.items()})
        else:  # 返回单个结果
            hooks[name]['input'] = input

        # module 层输出
        if isinstance(output, tuple) or isinstance(input, list):  # 某层返回多个结果
            hooks[name]['output'] = [out for out in output]
        elif isinstance(input, OrderedDict):
            hooks[name]['input'] = {k: v for k, v in input.items()}
        else:  # 返回单个结果
            hooks[name]['output'] = output

    return hook


def see_hook_fpn():
    my_input = torch.rand((1, 3, 480, 640))
    model = get_model('res50', (480, 640), 10, self_pretrained=True)
    hooks = {}
    model.backbone.fpn.register_forward_hook(get_hook('fpn', hooks))

    model.eval()  # 不用 training mode
    with torch.no_grad():
        res = model(my_input)
        # fpn 输入 多个 layer 不同 scale 的输出
        fpn_in = hooks['fpn']['input']  # list[OrderedDict]
        fpn_out = hooks['fpn']['output']  # OrderedDict

        print('fpn in')
        for idx, inp in enumerate(fpn_in):
            if isinstance(inp, OrderedDict):
                for k, v in inp.items():
                    print(k, v.size())
        print('fpn out')
        for k, v in fpn_out.items():
            print(k, v.size())

        """ 
        - fpn in (resnet backbone out)
        0 torch.Size([1, 256, 120, 160])    # layer 1, 1/4
        1 torch.Size([1, 512, 60, 80])      # layer 2, 1/8
        2 torch.Size([1, 1024, 30, 40])     # layer 3, 1/16
        3 torch.Size([1, 2048, 15, 20])     # layer 4, 1/32
        
        - BackboneWithFPN() 传入的 in_channels_list, out_channels
        in_channels_stage2 = 256
        in_channels_list = [
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = 256  # 统一为 256，共用 FPN 之后网络
        
        - FPN 内部
        (inner_blocks) 1x1 conv 将不同 in_channels 输入都转成 256
            process: 最近邻上采样，加和，得到 4 种 scale Features Pyramid
        (layer_blocks) 3x3 conv 将加和后的 features 再用 3x3 conv
        
        - fpn out 得到不同 scale 的 features，交给 rpn 生成 proposals
        0 torch.Size([1, 256, 120, 160])
        1 torch.Size([1, 256, 60, 80])
        2 torch.Size([1, 256, 30, 40])
        3 torch.Size([1, 256, 15, 20])
        pool torch.Size([1, 256, 8, 10])
        
        - roi_heads
          对 proposals 进行 MultiScaleRoIAlign, 得到统一大小的 feature
          box_head 2 个 fc, box_predictor 得到 cls_logits, box_regression
        """


if __name__ == '__main__':
    see_hook_fpn()
