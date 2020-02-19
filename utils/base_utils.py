import random
import numpy as np


def sub_two_list(l1, l2):
    assert len(l1) > len(l2), 'you may place 2 args in wrong positions'
    return list(set(l1) - set(l2))


if __name__ == '__main__':
    pass
