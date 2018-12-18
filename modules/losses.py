from torch.nn import functional
import torch


def mse_loss(x, x_rec, info):
    """MSE loss. For generality returns a dictionary."""
    x = x.reshape((x.shape[0], -1))
    x_rec = x_rec.reshape((x_rec.shape[0], -1))
    mse = torch.sum((x - x_rec) ** 2, dim=1)
    mse = torch.mean(mse, dim=0)
    return {'mse': mse}


def cross_entropy_loss(x, x_rec, info):
    """Cross entropy loss. For generality returns a dictionary."""
    x = x.reshape((x.shape[0], -1))
    x_rec = x_rec.reshape((x_rec.shape[0], -1))
    ce = functional.binary_cross_entropy(x_rec, x, reduction='none')
    ce = torch.sum(ce, dim=1)
    ce = torch.mean(ce, dim=0)
    return {'ce': ce}


def vae_mse_loss(x, x_rec, info):
    """Returns reconstruction (mse) loss and KL divergence."""
    mse = mse_loss(x, x_rec, info)['mse']
    z, mu, log_var = info
    kl_loss = -0.5 * torch.sum(1 + log_var - (mu ** 2) - torch.exp(log_var), dim=1)
    kl_loss = torch.mean(kl_loss, dim=0)
    return {'mse': mse, 'KL': kl_loss}


def vae_ce_loss(x, x_rec, info):
    """Returns reconstruction (ce) loss and KL divergence."""
    ce = cross_entropy_loss(x, x_rec, info)['ce']
    z, mu, log_var = info
    kl_loss = -0.5 * torch.sum(1 + log_var - (mu ** 2) - torch.exp(log_var), dim=1)
    kl_loss = torch.mean(kl_loss, dim=0)
    return {'ce': ce, 'KL': kl_loss}


def get_loss_function(architecture_args):
    loss_type = architecture_args.get('loss', 'mse')
    if loss_type == 'mse':
        return mse_loss
    if loss_type == 'ce':
        return cross_entropy_loss
    if loss_type == 'vae_mse':
        return vae_mse_loss
    if loss_type == 'vae_ce':
        return vae_ce_loss
    return None
