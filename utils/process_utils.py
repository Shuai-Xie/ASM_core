import numpy as np


def lasso_shift(records):
    mean = np.mean(records)
    return np.mean([abs(x - mean) for x in records])


def ridge_shift(records):
    mean = np.mean(records)  # = np.var(records)
    return np.mean([(x - mean) ** 2 for x in records])


def demo_test():
    demo_records = [
        [0.732, 0.74, 0.738],  # 0.0031
        [0.73, 0.74, 0.75],  # 0.0067
        [0.75, 0.752, 0.754]  # 0.0013
    ]

    for records in demo_records:
        print(lasso_shift(records))
        print(ridge_shift(records))
        print()


if __name__ == '__main__':
    demo_test()
