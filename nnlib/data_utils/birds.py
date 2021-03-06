from torchvision import transforms
import torch

from .base import log_call_parameters
from .abstract import StandardVisionDataset
from .torch_extensions import birds


class Birds(StandardVisionDataset):
    @log_call_parameters
    def __init__(self, data_augmentation: bool = False, **kwargs):
        super(Birds, self).__init__(**kwargs)
        self.data_augmentation = data_augmentation

    @property
    def dataset_name(self) -> str:
        return "birds"

    @property
    def means(self):
        return torch.tensor([0.485, 0.456, 0.406])

    @property
    def stds(self):
        return torch.tensor([0.229, 0.224, 0.225])

    @property
    def train_transforms(self):
        if not self.data_augmentation:
            return self.test_transforms

        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize_transform,
        ])

    @property
    def test_transforms(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            self.normalize_transform,
        ])

    def raw_dataset(self, data_dir: str, download: bool, train: bool, transform):
        return birds.Birds(data_dir, train=train, download=download, transform=transform)
