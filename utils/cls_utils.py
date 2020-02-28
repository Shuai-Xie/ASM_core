import numpy as np


def get_samples_per_class(targets):
    cnts = np.bincount(targets)
    min_, max_ = np.min(targets), np.max(targets)
    return {
        val: cnts[val] for val in range(min_, max_ + 1)
    }


if __name__ == '__main__':
    random_idxs = np.random.permutation(range(100))[:10]
    print(random_idxs)

    pass
