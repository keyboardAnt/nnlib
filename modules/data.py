from torchvision import datasets
import numpy as np
import os


def split(data, labels, val_percent=20.0):
    """ Split training set into train and validation sets.
    """
    train_cnt = int((1 - val_percent / 100.0) * data.shape[0])
    return data[:train_cnt], labels[:train_cnt], data[train_cnt:], labels[train_cnt:]


def load_mnist(val_percent=20.0):
    """ Load MNIST, split into train/val/test.
    Returns : train_data, train_labels, val_data, val_labels, test_data, test_labels
              train_data, val_data, test_data have shape (?, 1, 784)
    """
    data_dir = os.path.join(os.path.dirname(__file__), '../data/MNIST/')
    mnist_train = datasets.MNIST(data_dir, download=True, train=True)
    mnist_test = datasets.MNIST(data_dir, download=True, train=False)

    # normalize
    train_data = mnist_train.train_data.reshape((-1, 1, 28, 28)).numpy().astype(np.float32) / 256.0
    train_labels = mnist_train.train_labels.numpy().astype(np.int32)

    test_data = mnist_test.test_data.reshape((-1, 1, 28, 28)).numpy().astype(np.float32) / 256.0
    test_labels = mnist_test.test_labels.numpy().astype(np.int32)

    return split(train_data, train_labels, val_percent) + (test_data, test_labels)
