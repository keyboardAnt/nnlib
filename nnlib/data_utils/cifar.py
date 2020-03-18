from .base import split
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import numpy as np
import os
import torch


def load_cifar_datasets(val_ratio=0.2, data_augmentation=False, num_train_examples=None,
                        n_classes=10, seed=42, data_dir=None):

    if data_dir is None and n_classes == 10:
        data_dir = os.path.join(os.environ['DATA_DIR'], 'cifar10')
    if data_dir is None and n_classes == 100:
        data_dir = os.path.join(os.environ['DATA_DIR'], 'cifar100')
    assert data_dir is not None

    data_augmentation_transforms = []
    if data_augmentation:
        data_augmentation_transforms = [transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(32, 4)]

    # Add normalization. This is done so that models pretrained on ImageNet work well.
    means = torch.tensor([0.485, 0.456, 0.406])
    stds = torch.tensor([0.229, 0.224, 0.225])
    normalize_transform = transforms.Normalize(mean=means, std=stds)
    common_transforms = [transforms.ToTensor(), normalize_transform]

    train_transform = transforms.Compose(data_augmentation_transforms + common_transforms)
    val_transform = transforms.Compose(common_transforms)

    if n_classes == 10:
        dataset_class = datasets.CIFAR10
    else:
        dataset_class = datasets.CIFAR100

    train_data = dataset_class(data_dir, download=True, train=True, transform=train_transform)
    val_data = dataset_class(data_dir, download=True, train=True, transform=val_transform)
    test_data = dataset_class(data_dir, download=True, train=False, transform=val_transform)

    # split train and validation
    train_indices, val_indices = split(len(train_data), val_ratio, seed)
    if num_train_examples is not None:
        train_indices = np.random.choice(train_indices, num_train_examples, replace=False)
    train_data = Subset(train_data, train_indices)
    val_data = Subset(val_data, val_indices)

    # name datasets and save statistics
    for dataset in [train_data, val_data, test_data]:
        dataset.dataset_name = ('cifar10' if n_classes == 10 else 'cifar100')
        dataset.statistics = (means, stds)

    return train_data, val_data, test_data, None


def load_cifar_loaders(val_ratio=0.2, batch_size=128,  seed=42, drop_last=False, num_train_examples=None,
                       data_augmentation=False, n_classes=10):
    train_data, val_data, test_data, info = load_cifar_datasets(val_ratio=val_ratio,
                                                                data_augmentation=data_augmentation,
                                                                num_train_examples=num_train_examples,
                                                                n_classes=n_classes,
                                                                seed=seed)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=4, drop_last=drop_last)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                            num_workers=4, drop_last=drop_last)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                             num_workers=4, drop_last=drop_last)

    return train_loader, val_loader, test_loader, info
