from torch.utils.data import Dataset
from PIL import Image


class CIFAR(Dataset):
    def __init__(self, images, targets,
                 transform=None, target_transform=None):
        self.images = images
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img, target = self.images[index], self.targets[index]
        img = Image.fromarray(img)  # cvt to PIL

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
