from abc import abstractmethod

from torchvision import transforms, datasets
import torch

from .abstract import StandardVisionDataset
from .noise_tools import get_uniform_error_corruption_fn


class MNIST(StandardVisionDataset):
    def __init__(self, data_augmentation: bool = False):
        super(MNIST, self).__init__()
        self.data_augmentation = data_augmentation

    @property
    def dataset_name(self) -> str:
        return "mnist"

    @property
    def means(self):
        return torch.tensor([0.456])

    @property
    def stds(self):
        return torch.tensor([0.224])

    @property
    def train_transforms(self):
        if not self.data_augmentation:
            return self.test_transforms
        return transforms.Compose([transforms.RandomCrop(28, 4),
                                   transforms.ToTensor(),
                                   self.normalize_transform])

    @property
    def test_transforms(self):
        return transforms.Compose([
            transforms.ToTensor(),
            self.normalize_transform
        ])

    def raw_dataset(self, data_dir: str, download: bool, train: bool, transform):
        return datasets.MNIST(data_dir, download=download, train=train, transform=transform)


class NoisyMNIST(MNIST):
    def __init__(self, data_augmentation: bool = False, clean_validation: bool = False):
        super(NoisyMNIST, self).__init__(data_augmentation=data_augmentation)
        self.clean_validation = clean_validation

    @abstractmethod
    def corruption_fn(self, dataset):
        raise NotImplementedError('corruption_fn is not implemented')

    def post_process_datasets(self, train_data, val_data, test_data, info=None):
        is_corrupted = self.corruption_fn(train_data)
        if not self.clean_validation:
            _ = self.corruption_fn(val_data)
        return train_data, val_data, test_data, is_corrupted


class UniformNoiseMNIST(NoisyMNIST):
    def __init__(self, error_prob: float, data_augmentation: bool = False, clean_validation: bool = False):
        super(UniformNoiseMNIST, self).__init__(data_augmentation=data_augmentation,
                                                clean_validation=clean_validation)
        self._corruption_fn = get_uniform_error_corruption_fn(error_prob=error_prob, n_classes=10)

    def corruption_fn(self, dataset):
        return self._corruption_fn(dataset)
