from abc import abstractmethod, ABC
from torch.utils.data import Subset, DataLoader
from torchvision import transforms

import os
import numpy as np


def get_split_indices(n_samples, val_ratio, seed):
    np.random.seed(seed)
    train_cnt = int((1 - val_ratio) * n_samples)
    perm = np.random.permutation(n_samples)
    train_indices = perm[:train_cnt]
    val_indices = perm[train_cnt:]
    return train_indices, val_indices


def revert_normalization(samples, dataset):
    """ Reverts normalization of images.
    :param samples: 3D or 4D tensor of images.
    :param dataset: should have attribute `statistics`, which is a pair (means, stds)
    :return:
    """
    means, stds = dataset.statistics
    means = means.to(samples.device)
    stds = stds.to(samples.device)
    if len(samples.shape) == 3:
        samples = samples.unsqueeze(dim=0)
    return (samples * stds.unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3) +
            means.unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3))


def get_loaders_from_datasets(train_data, val_data, test_data, batch_size=128, num_workers=4, drop_last=False):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=drop_last)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=drop_last)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, drop_last=drop_last)
    return train_loader, val_loader, test_loader


def print_dataset_info_decorator(build_loaders):
    def wrapper(*args, **kwargs):
        train_loader, val_loader, test_loader, info = build_loaders(*args, **kwargs)
        example_shape = train_loader.dataset[0][0].shape
        print("Dataset {} is loaded".format(train_loader.dataset.dataset_name))
        print("\ttrain_samples: {}".format(len(train_loader.dataset)))
        if val_loader is not None:
            print("\tval_samples: {}".format(len(val_loader.dataset)))
        if test_loader is not None:
            print("\ttest_samples: {}".format(len(test_loader.dataset)))
        print("\texample_shape: {}".format(example_shape))
        return train_loader, val_loader, test_loader, info
    return wrapper


class StandardVisionDataset(ABC):
    """
    Holds information about a given dataset and implements several useful functions
    """

    def __init__(self):
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
    def train_transforms(self):
        return transforms.Compose([
            transforms.ToTensor(),
            self.normalize_transform
        ])

    @property
    def test_transforms(self):
        return self.train_transforms()

    def build_datasets(self, data_dir: str = None, val_ratio: float = 0.2, num_train_examples: int = None,
                       seed: int = 42):
        """ Builds train, validation, and test datasets. """

        if data_dir is None:
            data_dir = os.path.join(os.environ['DATA_DIR'], self.dataset_name)

        train_data = self.raw_dataset(data_dir, download=True, train=True, transform=self.train_transforms)
        val_data = self.raw_dataset(data_dir, download=True, train=True, transform=self.train_transforms)
        test_data = self.raw_dataset(data_dir, download=True, train=False, transform=self.test_transforms)

        # split train and validation
        train_indices, val_indices = get_split_indices(len(train_data), val_ratio, seed)
        if num_train_examples is not None:
            train_indices = np.random.choice(train_indices, num_train_examples, replace=False)
        train_data = Subset(train_data, train_indices)
        val_data = Subset(val_data, val_indices)

        # name datasets and save statistics
        for dataset in [train_data, val_data, test_data]:
            dataset.dataset_name = self.dataset_name
            dataset.statistics = (self.means, self.stds)

        # general way of returning extra information
        info = None

        return train_data, val_data, test_data, info

    @print_dataset_info_decorator
    def build_loaders(self, data_dir: str = None, val_ratio: float = 0.2, num_train_examples: int = None,
                      seed: int = 42, batch_size: int = 128, num_workers: int = 4, drop_last: bool = False):
        train_data, val_data, test_data, info = self.build_datasets(data_dir=data_dir, val_ratio=val_ratio,
                                                                    num_train_examples=num_train_examples,
                                                                    seed=seed)
        train_loader, val_loader, test_loader = get_loaders_from_datasets(train_data, val_data, test_data,
                                                                          batch_size=batch_size,
                                                                          num_workers=num_workers,
                                                                          drop_last=drop_last)
        return train_loader, val_loader, test_loader, info


def load_data_from_arguments(args):
    """ Helper method for loading data from arguments.
    """
    if args.dataset == 'mnist':
        from .mnist import MNIST
        return MNIST(data_augmentation=args.data_augmentation).build_loaders(
            batch_size=args.batch_size, num_train_examples=args.num_train_examples, seed=args.seed)

    if args.dataset == 'fashion-mnist':
        from .fashion_mnist import FashionMNIST
        return FashionMNIST(data_augmentation=args.data_augmentation).build_loaders(
            batch_size=args.batch_size, num_train_examples=args.num_train_examples, seed=args.seed)

    if args.dataset == 'cifar10':
        from .cifar import CIFAR
        return CIFAR(num_classes=10, data_augmentation=args.data_augmentation).build_loaders(
            batch_size=args.batch_size, num_train_examples=args.num_train_examples, seed=args.seed)

    if args.dataset == 'cifar100':
        from .cifar import CIFAR
        return CIFAR(num_classes=100, data_augmentation=args.data_augmentation).build_loaders(
            batch_size=args.batch_size, num_train_examples=args.num_train_examples, seed=args.seed)

    if args.dataset == 'imagenet':
        from .imagenet import ImageNet
        return ImageNet(data_augmentation=args.data_augmentation).build_loaders(
            batch_size=args.batch_size, num_train_examples=args.num_train_examples, seed=args.seed)

    if args.dataset == 'dsprites':
        from .dsprites import load_dsprites_loaders
        return load_dsprites_loaders(batch_size=args.batch_size, seed=args.seed, colored=args.colored)

    raise ValueError("args.dataset is not recognized")
