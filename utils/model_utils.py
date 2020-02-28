import torch
from operator import mul
from functools import reduce
from torchsummary import summary


def get_list_mul(li):
    return reduce(mul, li) if len(li) > 0 else 0


def print_model_summary(model, input_size):
    summary(model, input_size, device='cpu')


def print_model_params(model):
    """
    model.parameters(): typically passed to an optimizer. 与下面相比，只少了 name
        param type: torch.nn.parameter.Parameter, Parameter(torch.Tensor)
    """
    print('=> model params:')
    k = 0
    print('{:4} {:30} {:10} {}'.format('idx', 'size', 'params', 'grad'))
    for idx, param in enumerate(model.parameters()):  # generator
        print('{:<4} {:30} '.format(idx, str(param.size())), end='')
        multi = get_list_mul(param.size())  # [out_C, in_C, kernel_w, kernel_h]
        print('{:<10} {}'.format(multi, param.requires_grad))
        k += multi
    print('total params:', k)


def print_model_named_params(model):
    """
    model.named_parameters():
        yielding both the name of the parameter as well as the parameter itself
        print, :< 左对齐, :> 右对齐; 默认情况 str 左对齐，number 右对齐
    """
    print('=> model named params:')
    print('{:4} {:50} {:30} {:10} {}'.format('idx', 'name', 'size', 'params', 'grad'))
    for idx, (name, param) in enumerate(model.named_parameters()):  # generator, 生成 name, param tensor
        print('{:<4} {:50} {:30} {:<10} {}'.format(
            idx, name, str(param.size()), get_list_mul(param.size()), param.requires_grad))


def print_model_state_dict(model):
    """
    model.state_dict(): a dictionary containing a whole state of the module.
        Both parameters and persistent buffers (e.g. running averages)
        are included. Keys are corresponding parameter and buffer names.
    """
    state_dict = model.state_dict()
    print('{:4} {:50} {:30} {:10} {}'.format('idx', 'name', 'size', 'params', 'grad'))
    for idx, (name, param) in enumerate(state_dict.items()):
        print('{:<4} {:50} {:30} {:<10} {}'.format(
            idx, name, str(param.size()), get_list_mul(param.size()), param.requires_grad))


def load_model(model, ckpt_path, optimizer=None):
    """
    model 和 optimizer 从 ckpt_path load_state_dict
    """
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['state_dict'])
    print('load {}, epoch {}'.format(ckpt_path, ckpt['epoch']))

    best_epoch = ckpt.get('epoch', -1)
    best_acc = ckpt.get('accuracy', 0)

    if optimizer is not None:
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        return model, optimizer, best_epoch, best_acc
    else:
        return model


def save_model(ckpt_path, model, epoch, accuracy, optimizer=None):
    # model = { 'epoch': , 'state_dict': , 'optimizer' }
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()  # convert data_parallal to model
    else:
        state_dict = model.state_dict()
    ckpt = {
        'epoch': epoch,
        'accuracy': accuracy,
        'state_dict': state_dict
    }
    if optimizer is not None:
        ckpt['optimizer'] = optimizer.state_dict()
    torch.save(ckpt, ckpt_path)
    print('save {}, epoch {}'.format(ckpt_path, ckpt['epoch']))


if __name__ == '__main__':
    # model = resnet18(pretrained=False)
    # print_model_state_dict(model)
    # print_model_named_params(model)
    # print(model)

    pass
