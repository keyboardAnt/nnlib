from .base import StandardVisionDataset
from torchvision import transforms, datasets

import torch


class CIFAR(StandardVisionDataset):
    def __init__(self, num_classes: int = 10, data_augmentation: bool = False):
        super(CIFAR, self).__init__()
        assert num_classes in [10, 100]
        self.num_classes = num_classes
        self.data_augmentation = data_augmentation

    @property
    def dataset_name(self) -> str:
        if self.num_classes == 10:
            return "cifar10"
        if self.num_classes == 100:
            return "cifar100"
        raise ValueError("num_classes should be 10 or 100")

    @property
    def means(self):
        return torch.tensor([0.485, 0.456, 0.406])

    @property
    def stds(self):
        return torch.tensor([0.229, 0.224, 0.225])

    @property
    def train_transforms(self):
        data_augmentation_transforms = []
        if self.data_augmentation:
            data_augmentation_transforms = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4)]
        return transforms.Compose([transforms.ToTensor()]
                                  + data_augmentation_transforms
                                  + [self.normalize_transform])

    @property
    def test_transforms(self):
        return transforms.Compose([
            transforms.ToTensor(),
            self.normalize_transform
        ])

    def raw_dataset(self, data_dir: str, download: bool, train: bool, transform):
        if self.num_classes == 10:
            return datasets.CIFAR10(data_dir, download=download, train=train, transform=transform)
        if self.num_classes == 100:
            return datasets.CIFAR100(data_dir, download=download, train=train, transform=transform)
        raise ValueError("num_classes should be 10 or 100")
