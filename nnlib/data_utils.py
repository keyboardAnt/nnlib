from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import numpy as np
import os
import torch


def split(n_samples, val_ratio, seed):
    train_cnt = int((1 - val_ratio) * n_samples)
    np.random.seed(seed)
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


def load_mnist_datasets(val_ratio=0.2, num_train_examples=None, seed=42):
    data_dir = os.path.join(os.path.dirname(__file__), '../data/mnist/')

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
    train_data, val_data, test_data, _ = load_mnist_datasets(
        val_ratio=val_ratio, num_train_examples=num_train_examples, seed=seed)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=4, drop_last=drop_last)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                            num_workers=4, drop_last=drop_last)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                             num_workers=4, drop_last=drop_last)

    return train_loader, val_loader, test_loader


def load_cifar_datasets(val_ratio=0.2, data_augmentation=False, num_train_examples=None, n_classes=10, seed=42):
    if n_classes == 10:
        data_dir = os.path.join(os.path.dirname(__file__), '../data/cifar10/')
    elif n_classes == 100:
        data_dir = os.path.join(os.path.dirname(__file__), '../data/cifar100/')
    else:
        raise ValueError("Variable n_classes should be 10 or 100.")

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
    train_data, val_data, test_data, _ = load_cifar_datasets(val_ratio=val_ratio,
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

    return train_loader, val_loader, test_loader


def load_data_from_arguments(args):
    """ Helper method for loading data from arguments.
    """
    if args.dataset == 'mnist':
        train_loader, val_loader, test_loader = load_mnist_loaders(
            batch_size=args.batch_size, num_train_examples=args.num_train_examples,
            seed=args.seed)

    if args.dataset == 'cifar10':
        train_loader, val_loader, test_loader = load_cifar_loaders(
            batch_size=args.batch_size,
            num_train_examples=args.num_train_examples,
            data_augmentation=args.data_augmentation,
            n_classes=10,
            seed=args.seed)

    if args.dataset == 'cifar100':
        train_loader, val_loader, test_loader = load_cifar_loaders(
            batch_size=args.batch_size,
            num_train_examples=args.num_train_examples,
            data_augmentation=args.data_augmentation,
            n_classes=100,
            seed=args.seed)

    example_shape = train_loader.dataset[0][0].shape
    print("Dataset is loaded:\n\ttrain_samples: {}\n\tval_samples: {}\n\t"
          "test_samples: {}\n\tsample_shape: {}".format(
        len(train_loader.dataset), len(val_loader.dataset),
        len(test_loader.dataset), example_shape))

    return train_loader, val_loader, test_loader
