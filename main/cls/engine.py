import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from datasets.cls_datasets import CIFAR
from torch.utils.data import DataLoader


def train_one_epoch(model, dataloader, device, epoch,
                    writer, optimizer, criterion):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    begin_step = epoch * len(dataloader)
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)  # logtis?
        outputs = F.softmax(outputs, dim=-1)
        loss = criterion(outputs, targets)  # (N,C),(N) 交叉熵定义
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)  # pred results
        correct += predicted.eq(targets).sum().item()  # eq=1，直接 sum
        total += targets.size(0)  # total samples

        # save mean loss of each batch
        writer.add_scalar('Train/loss', train_loss / (batch_idx + 1), global_step=begin_step + batch_idx)

    train_acc = correct / total
    return train_acc


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    eval_acc = correct / total
    return eval_acc


@torch.no_grad()
def detect_unlabel_imgs(model, batch_unlabel_imgs, batch_unlabel_targets, device):
    model.eval()
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # pass targets to build dataloader
    dataset = CIFAR(batch_unlabel_imgs, batch_unlabel_targets, transform=transform_test)
    dataloader = DataLoader(dataset, batch_size=100,
                            shuffle=False, num_workers=4, drop_last=False)

    y_pred_prob = torch.empty((0, 10)).to(device)

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=-1)  # logits -> probs [100,10]
        y_pred_prob = torch.cat((y_pred_prob, probs), dim=0)

    return y_pred_prob.cpu().numpy()  # to np


class Delta_Scheduler:
    def __init__(self, init_val, min_val, max_steps):
        self.delta = init_val
        self.step_decay = (init_val - min_val) / max_steps

    def step(self):
        self.delta -= self.step_decay
