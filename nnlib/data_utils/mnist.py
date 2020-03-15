from .base import split
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import os
import torch
import numpy as np


def load_mnist_datasets(val_ratio=0.2, num_train_examples=None, seed=42, data_dir=None):

    if data_dir is None:
        data_dir = os.path.join(os.environ['DATA_DIR'], 'mnist')

    # Add normalization. This is done so that models pretrained on ImageNet work well.
    means = torch.tensor([0.456])
    stds = torch.tensor([0.224])
    normalize_transform = transforms.Normalize(mean=means, std=stds)

    composed_transform = transforms.Compose([transforms.ToTensor(), normalize_transform])

    train_val_data = datasets.MNIST(data_dir, download=True, train=True, transform=composed_transform)
    test_data = datasets.MNIST(data_dir, download=True, train=False, transform=composed_transform)

    # split train and validation
    train_indices, val_indices = split(len(train_val_data), val_ratio, seed)
    if num_train_examples is not None:
        train_indices = np.random.choice(train_indices, num_train_examples, replace=False)
    train_data = Subset(train_val_data, train_indices)
    val_data = Subset(train_val_data, val_indices)

    # name datasets and save statistics
    for dataset in [train_data, val_data, test_data]:
        dataset.dataset_name = 'mnist'
        dataset.statistics = (means, stds)

    return train_data, val_data, test_data, None


def load_mnist_loaders(val_ratio=0.2, batch_size=128, seed=42, drop_last=False, num_train_examples=None):
    train_data, val_data, test_data, info = load_mnist_datasets(
        val_ratio=val_ratio, num_train_examples=num_train_examples, seed=seed)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=4, drop_last=drop_last)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                            num_workers=4, drop_last=drop_last)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                             num_workers=4, drop_last=drop_last)

    return train_loader, val_loader, test_loader, info
