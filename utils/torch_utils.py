import torch


def get_gpu_names():
    gpu_names = []
    for gpu_id in range(torch.cuda.device_count()):
        dev_name = torch.cuda.get_device_name(gpu_id)
        gpu_names.append(dev_name)
    print(gpu_names)  # ['Quadro M6000 24GB', 'Quadro K5200']
    return gpu_names


def get_cpu_threads():
    # CPU threads
    n_threads = torch.get_num_threads()  # 8
    print(n_threads)
    return n_threads


def torch_where():
    a = torch.tensor([1, 2, 3, 4])
    b = a.where(a % 2 == 0, torch.tensor(0))  # 参数 other，表示不满足的元素填充值
    c = torch.where(a % 2 == 0, a, torch.tensor(-1))
    print(b)
    print(c)


def rand_tensor():
    # To increase the reproducibility of result, we often set the random seed to a specific value first.
    # 设置 seed，rand 结果可复现
    torch.manual_seed(1)
    v = torch.rand(2, 3)  # Initialize with random number (uniform distribution)
    v = torch.randn(2, 3)  # With normal distribution (SD=1, mean=0)
    v = torch.randperm(4)  # Size 4. Random permutation of integers from 0 to 3


def select_items():
    a = torch.arange(1, 10)  # torch.int64
    mask = [2, 3, 4]
    b = a[mask]
    print(b)
    mask = torch.tensor(mask)
    unmask = [idx for idx in range(a.size()[0]) if idx not in mask]
    b = a[unmask]
    print(b)

    a = torch.arange(0, 10)
    b = torch.arange(10, 20)
    # True 返回 1， False 返回 0，通过 nonzero() 得到下标
    mask = torch.where(a % 2 == 0, torch.tensor(1), torch.tensor(0)).nonzero()
    print(mask)  # 获得下标
    print(mask.size()[0])
    print(b[mask])

    mask = torch.gt(a, 1)
    print(mask)
    print(b[mask])


def top_k_idx_val():
    torch.manual_seed(2)  # 固定了 seed 后，每次得到的 a 都一样
    a = torch.randperm(10)
    print(a)
    idxs = torch.where(a > 2, torch.tensor(1), torch.tensor(0)).nonzero()
    print(idxs)
    val, idx, = torch.topk(a, k=1, dim=-1)
    print(idx, val)


def demo_clone():
    anns = torch.tensor([1, 2, 3, 5]).nonzero().reshape(-1)
    print(anns)
    print(anns.size())
    exit(0)
    a = anns
    b = anns.clone()  # 保存原值
    a[2] = 100
    print(a)
    print(b)


if __name__ == '__main__':
    # top_k_idx_val()
    demo_clone()

    pass
