from torch.utils.data import DataLoader
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


def register_parser(registry, parser_name):
    def decorator(parser_fn):
        registry[parser_name] = parser_fn

        def wrapper(*args, **kwargs):
            return parser_fn(*args, **kwargs)

        return wrapper

    return decorator


class DataSelector:
    """ Helper class for loading data from arguments. """

    _parsers = {}  # register_parsers decorator will fill this

    def __init__(self):
        pass

    @register_parser(_parsers, 'mnist')
    def _parse_mnist(self, args):
        from .mnist import MNIST
        return MNIST(data_augmentation=args.data_augmentation).build_loaders(
            batch_size=args.batch_size, num_train_examples=args.num_train_examples, seed=args.seed)

    @register_parser(_parsers, 'uniform-noise-mnist')
    def _parse_uniform_noise_mnist(self, args):
        from .mnist import UniformNoiseMNIST
        return UniformNoiseMNIST(error_prob=args.error_prob, clean_validation=args.clean_validation,
                                 data_augmentation=args.data_augmentation).build_loaders(
            batch_size=args.batch_size, num_train_examples=args.num_train_examples, seed=args.seed)

    @register_parser(_parsers, 'fashion-mnist')
    def _parse_fashion_mnist(self, args):
        from .fashion_mnist import FashionMNIST
        return FashionMNIST(data_augmentation=args.data_augmentation).build_loaders(
            batch_size=args.batch_size, num_train_examples=args.num_train_examples, seed=args.seed)

    @register_parser(_parsers, 'cifar10')
    def _parse_cifar10(self, args):
        from .cifar import CIFAR
        return CIFAR(n_classes=10, data_augmentation=args.data_augmentation).build_loaders(
            batch_size=args.batch_size, num_train_examples=args.num_train_examples, seed=args.seed)

    @register_parser(_parsers, 'uniform-noise-cifar10')
    def _parse_uniform_noise_cifar10(self, args):
        from .cifar import UniformNoiseCIFAR
        return UniformNoiseCIFAR(n_classes=10, error_prob=args.error_prob, data_augmentation=args.data_augmentation,
                                 clean_validation=args.clean_validation).build_loaders(
            batch_size=args.batch_size, num_train_examples=args.num_train_examples, seed=args.seed)

    @register_parser(_parsers, 'pair-noise-cifar10')
    def _parse_pair_noise_cifar10(self, args):
        from .cifar import PairNoiseCIFAR10
        return PairNoiseCIFAR10(error_prob=args.error_prob, data_augmentation=args.data_augmentation,
                                clean_validation=args.clean_validation).build_loaders(
            batch_size=args.batch_size, num_train_examples=args.num_train_examples, seed=args.seed)

    @register_parser(_parsers, 'cifar100')
    def _parse_cifar100(self, args):
        from .cifar import CIFAR
        return CIFAR(n_classes=100, data_augmentation=args.data_augmentation).build_loaders(
            batch_size=args.batch_size, num_train_examples=args.num_train_examples, seed=args.seed)

    @register_parser(_parsers, 'uniform-noise-cifar100')
    def _parse_uniform_noise_cifar100(self, args):
        from .cifar import UniformNoiseCIFAR
        return UniformNoiseCIFAR(n_classes=100, error_prob=args.error_prob, data_augmentation=args.data_augmentation,
                                 clean_validation=args.clean_validation).build_loaders(
            batch_size=args.batch_size, num_train_examples=args.num_train_examples, seed=args.seed)

    @register_parser(_parsers, 'imagenet')
    def _parse_imagenet(self, args):
        from .imagenet import ImageNet
        return ImageNet(data_augmentation=args.data_augmentation).build_loaders(
            batch_size=args.batch_size, num_train_examples=args.num_train_examples, seed=args.seed)

    @register_parser(_parsers, 'dsprites')
    def _parse_dsprites(self, args):
        from .dsprites import load_dsprites_loaders
        return load_dsprites_loaders(batch_size=args.batch_size, seed=args.seed, colored=args.colored)

    @register_parser(_parsers, 'clothing1m')
    def _parse_clothing1m(self, args):
        from .clothing1m import Clothing1M
        return Clothing1M(data_augmentation=args.data_augmentation).build_loaders(
            batch_size=args.batch_size, num_train_examples=args.num_train_examples,
            num_workers=10, seed=args.seed)

    def can_parse(self, dataset_name):
        return dataset_name in self._parsers

    def parse(self, args):
        if not self.can_parse(args.dataset):
            raise ValueError(f"Value {args.dataset} for args.dataset is not recognized")
        parser = self._parsers[args.dataset]
        return parser(self, args)
