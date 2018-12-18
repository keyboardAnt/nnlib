from modules import utils
import os
import torch


def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def convert_numpy(x):
    if x.requires_grad:
        x = x.detach()
    if x.device.type != 'cpu':
        x = x.cpu()
    return x.numpy()


def save(model, path, verbose=False):
    save_dir = os.path.dirname(path)
    utils.make_path(save_dir)
    if verbose:
        print('Saving into {}'.format(path))
    torch.save(model, path)


def load(path):
    return torch.load(path)
