import random


def sub_two_list(l1, l2):
    assert len(l1) > len(l2), 'you may place 2 args in wrong positions'
    return list(set(l1) - set(l2))


if __name__ == '__main__':
    l1 = [1, 2, 3, 4, 6, 8]
    l2 = [3, 4, 8]

