import random
import numpy as np
import re
import torch


def sub_two_list(l1, l2):
    assert len(l1) > len(l2), 'you may place 2 args in wrong positions'
    return list(set(l1) - set(l2))


def re_search():
    model = 'model_2.pth'
    epoch = re.search(r'model_(\d).pth', model).group(1)
    print(epoch)


def test_zip():
    a_list = [1, 2, 3, 4]
    b_list = [3, 4, 5, 6]

    for a, b in zip(a_list, b_list):
        print(a, b)


def np_size():
    a = np.random.rand(2, 3)
    print(a.size)  # 所有元素数量


def oranges():
    for x in range(11, 15):
        # 两个两个数多1个，三个三个数少1个
        if x % 2 == 1 and x % 3 == 2:
            print(x)
            break


if __name__ == '__main__':
    a = torch.tensor([1, 2, 3], dtype=torch.float32)
    a = torch.mean(a).item()
    print(a)
    print(type(a))

    pass
