import torch
import torch.nn.functional as F


def binary_cross_entropy(target, pred):
    target = target.reshape((target.shape[0], -1))
    pred = pred.reshape((pred.shape[0], -1))
    ce = F.binary_cross_entropy(pred, target, reduction='none')
    ce = torch.sum(ce, dim=1)
    ce = torch.mean(ce, dim=0)
    return ce


def mse(target, pred):
    target = target.reshape((target.shape[0], -1))
    pred = pred.reshape((pred.shape[0], -1))
    mse = torch.sum((target - pred) ** 2, dim=1)
    mse = torch.mean(mse, dim=0)
    return mse


def mae(target, pred):
    target = target.reshape((target.shape[0], -1))
    pred = pred.reshape((pred.shape[0], -1))
    mad = torch.sum(torch.abs(target - pred), dim=1)
    mad = torch.mean(mad, dim=0)
    return mad
