import numpy as np


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


def load_data_from_arguments(args):
    """ Helper method for loading data from arguments.
    """
    if args.dataset == 'mnist':
        from .mnist import load_mnist_loaders
        train_loader, val_loader, test_loader, info = load_mnist_loaders(
            batch_size=args.batch_size,
            num_train_examples=args.num_train_examples,
            seed=args.seed)

    if args.dataset == 'fashion_mnist':
        from .fashion_mnist import load_fashion_mnist_loaders
        train_loader, val_loader, test_loader, info = load_fashion_mnist_loaders(
            batch_size=args.batch_size,
            num_train_examples=args.num_train_examples,
            seed=args.seed)

    if args.dataset == 'cifar10':
        from .cifar import load_cifar_loaders
        train_loader, val_loader, test_loader, info = load_cifar_loaders(
            batch_size=args.batch_size,
            num_train_examples=args.num_train_examples,
            data_augmentation=args.data_augmentation,
            n_classes=10,
            seed=args.seed)

    if args.dataset == 'cifar100':
        from .cifar import load_cifar_loaders
        train_loader, val_loader, test_loader, info = load_cifar_loaders(
            batch_size=args.batch_size,
            num_train_examples=args.num_train_examples,
            data_augmentation=args.data_augmentation,
            n_classes=100,
            seed=args.seed)

    if args.dataset == 'dsprites':
        from .dsprites import load_dsprites_loaders
        train_loader, val_loader, test_loader, info = load_dsprites_loaders(
            batch_size=args.batch_size,
            seed=args.seed,
            colored=args.colored)

    example_shape = train_loader.dataset[0][0].shape
    print("Dataset is loaded:\n\ttrain_samples: {}\n\tval_samples: {}\n\t"
          "test_samples: {}\n\tsample_shape: {}".format(
        len(train_loader.dataset), len(val_loader.dataset),
        len(test_loader.dataset), example_shape))

    return train_loader, val_loader, test_loader, info
