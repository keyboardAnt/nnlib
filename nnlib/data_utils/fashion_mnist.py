from .base import StandardVisionDataset
from torchvision import transforms, datasets

import torch


class FashionMNIST(StandardVisionDataset):
    def __init__(self, data_augmentation: bool = False):
        super(FashionMNIST, self).__init__()
        self.data_augmentation = data_augmentation

    @property
    def dataset_name(self) -> str:
        return "fashion-mnist"

    @property
    def means(self):
        return torch.tensor([0.456])

    @property
    def stds(self):
        return torch.tensor([0.224])

    @property
    def train_transforms(self):
        data_augmentation_transforms = []
        if self.data_augmentation:
            data_augmentation_transforms = [transforms.RandomCrop(28, 4), transforms.RandomHorizontalFlip()]
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
        return datasets.FashionMNIST(data_dir, download=download, train=train, transform=transform)
