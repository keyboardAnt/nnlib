from abc import abstractmethod, ABC
from torch.utils.data import Subset
from torchvision import transforms
from .base import get_split_indices, print_loaded_dataset_shapes, get_loaders_from_datasets, log_call_parameters

import os
import numpy as np


class StandardVisionDataset(ABC):
    """
    Holds information about a given dataset and implements several useful functions
    """

    def __init__(self, **kwargs):
        pass

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        raise NotImplementedError('dataset_name not implemented')

    @property
    @abstractmethod
    def means(self):
        raise NotImplementedError('means not implemented')

    @property
    @abstractmethod
    def stds(self):
        raise NotImplementedError('stds not implemented')

    @abstractmethod
    def raw_dataset(self, data_dir: str, download: bool, train: bool, transform):
        raise NotImplementedError('raw_dataset_class not implemented, need to return datasets')

    @property
    def normalize_transform(self):
        return transforms.Normalize(mean=self.means, std=self.stds)

    @property
    def train_transforms(
            self,
            data_normalizing: bool = True
    ):
        sequential_transforms = [transforms.ToTensor()]
        if data_normalizing:
            sequential_transforms.append(self.normalize_transform)
        return transforms.Compose(sequential_transforms)

    @property
    def test_transforms(self):
        return self.train_transforms()

    def post_process_datasets(self, train_data, val_data, test_data, info=None):
        """ This can be used to modify the labels or images. """
        return train_data, val_data, test_data, info

    @print_loaded_dataset_shapes
    @log_call_parameters
    def build_datasets(self, data_dir: str = None, val_ratio: float = 0.2, num_train_examples: int = None,
                       seed: int = 42, download: bool = True, **kwargs):
        """ Builds train, validation, and test datasets. """
        if data_dir is None:
            data_dir = os.path.join(os.environ['DATA_DIR'], self.dataset_name)

        train_data = self.raw_dataset(data_dir, download=download, train=True, transform=self.train_transforms)
        val_data = self.raw_dataset(data_dir, download=download, train=True, transform=self.train_transforms)
        test_data = self.raw_dataset(data_dir, download=download, train=False, transform=self.test_transforms)

        # split train and validation
        train_indices, val_indices = get_split_indices(len(train_data), val_ratio, seed)
        if num_train_examples is not None:
            train_indices = np.random.choice(train_indices, num_train_examples, replace=False)
        train_data = Subset(train_data, train_indices)
        val_data = Subset(val_data, val_indices)

        # general way of returning extra information
        info = None

        # post-process datasets
        train_data, val_data, test_data, info = self.post_process_datasets(train_data, val_data, test_data, info=info)

        # name datasets and save statistics
        for dataset in [train_data, val_data, test_data]:
            dataset.dataset_name = self.dataset_name
            dataset.statistics = (self.means, self.stds)

        return train_data, val_data, test_data, info

    @log_call_parameters
    def build_loaders(self, data_dir: str = None, val_ratio: float = 0.2, num_train_examples: int = None,
                      seed: int = 42, download: bool = True, batch_size: int = 128, num_workers: int = 4,
                      drop_last: bool = False, **kwargs):
        train_data, val_data, test_data, info = self.build_datasets(data_dir=data_dir, val_ratio=val_ratio,
                                                                    num_train_examples=num_train_examples,
                                                                    seed=seed, download=download, **kwargs)
        train_loader, val_loader, test_loader = get_loaders_from_datasets(train_data, val_data, test_data,
                                                                          batch_size=batch_size,
                                                                          num_workers=num_workers,
                                                                          drop_last=drop_last)
        return train_loader, val_loader, test_loader, info
