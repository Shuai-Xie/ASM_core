import numpy as np


# select `n_samples` informative samples from `y_pred_prob`


# Random sampling
def random_sampling(y_pred_prob, n_samples):
    return np.random.choice(range(len(y_pred_prob)), n_samples)


# Rank all the unlabeled samples in an ascending order according to the least confidence
def least_confidence(y_pred_prob, n_samples):
    """
    @param y_pred_prob: model outputs, (N,10) [logits, prob] 均可
    @param n_samples: choose number
    @return:
        lci: [ori_idx, prob, label]  # (N,3)
        lci_idx: [ori_idx]  # (N,)
    """
    origin_index = np.arange(0, len(y_pred_prob))
    max_prob = np.max(y_pred_prob, axis=1)
    pred_label = np.argmax(y_pred_prob, axis=1)

    lci = np.column_stack((origin_index, max_prob, pred_label))
    lci = lci[lci[:, 1].argsort()]  # 按照 prob 排序
    return lci[:n_samples], lci[:, 0].astype(int)[:n_samples]


# Rank all the unlabeled samples in an ascending order according to the margin sampling
def margin_sampling(y_pred_prob, n_samples):
    origin_index = np.arange(0, len(y_pred_prob))
    margim_sampling = np.diff(-np.sort(y_pred_prob)[:, ::-1][:, :2])
    pred_label = np.argmax(y_pred_prob, axis=1)

    msi = np.column_stack((origin_index, margim_sampling, pred_label))
    msi = msi[msi[:, 1].argsort()]
    return msi[:n_samples], msi[:, 0].astype(int)[:n_samples]


# Rank all the unlabeled samples in an descending order according to their entropy
def entropy(y_pred_prob, n_samples):
    origin_index = np.arange(0, len(y_pred_prob))
    entro = -np.nansum(np.multiply(y_pred_prob, np.log(y_pred_prob)), axis=1)
    pred_label = np.argmax(y_pred_prob, axis=1)

    eni = np.column_stack((origin_index, entro, pred_label))
    eni = eni[(-eni[:, 1]).argsort()]
    return eni[:n_samples], eni[:, 0].astype(int)[:n_samples]


# Choose select criterion function
def get_select_criterion_fn(criterion):
    if criterion == 'lc':
        fn = least_confidence
    elif criterion == 'ms':
        fn = margin_sampling
    elif criterion == 'en':
        fn = entropy
    else:
        raise ValueError('no such criterion')
    return fn


# Rank high confidence samples by entropy
def get_high_confidence_samples(y_pred_prob, delta):
    eni, eni_idx = entropy(y_pred_prob, len(y_pred_prob))
    hcs = eni[eni[:, 1] < delta]
    return hcs[:, 0].astype(int), hcs[:, 2].astype(int)


def see_cross_entroy():
    # prob = np.array([0.9991] + [0.0001] * 9) # 0.00918890121322387
    # prob = np.array([0.991] + [0.001] * 9)  # 0.07112917546111895
    prob = np.array([0.91] + [0.01] * 9)  # 0.5002880350577578
    print(np.sum(prob))
    entro = -np.nansum(np.multiply(prob, np.log(prob)), axis=0)  # nan -> 0
    print(entro)


if __name__ == '__main__':
    see_cross_entroy()
