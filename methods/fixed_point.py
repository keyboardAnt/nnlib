""" Auto-encoder fixed point experiments

Conventions
    ----------
    Code follows sklearn naming/style (e.g. fit(X) to train, transform() to apply model to test data,
    predict() recovers inputs from latent factors.

Code below by:
Greg Ver Steeg (gregv@isi.edu), 2018.
"""
from tensorboardX import SummaryWriter
from collections import defaultdict
from modules import nn, losses, utils, training
import os
import time
import numpy as np
import torch


class AE(object):
    """
    Fixed Point Auto-Encoder.
    Parameters
    ----------
    architecture_args : dictionary of arguments to pass to architecture building routine.

    training_args : contains arguments for building optimizer and LR scheduler.
    """

    def __init__(self, architecture_args, training_args, batch_size=128, epochs=100,
                 save_iter=10, device='cuda', log_dir=None, verbose=False):
        self.architecture_args = architecture_args
        self.training_args = training_args
        self.batch_size = batch_size
        self.epochs = epochs
        self.save_iter = save_iter
        self.device = device

        # if log_dir is not given, logging will be done a new directory in 'logs/' directory
        if log_dir is None:
            log_root = os.path.join(os.path.abspath(os.curdir), 'logs/')
            utils.make_path(log_root)
            last_run = max([0] + [int(k) for k in os.listdir(log_root) if k.isdigit()])
            log_dir = os.path.join(log_root, '{0:04d}'.format(last_run + 1))
            utils.make_path(log_dir)
        if verbose:
            print("Visualize logs using: tensorboard --logdir={0}".format(log_dir))

        self.log_dir = log_dir
        self.verbose = verbose

        # Initialize these when we fit on data
        self.network = None
        self.loss_function = None
        self.optimizer = None
        self.scheduler = None

    def transform(self, x):
        """ Get latent factors. Sampling is disabled. """
        self.network.eval()
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float, device=self.device)
            n_samples = x.shape[0]
            z = []
            for i in range(0, x.shape[0], self.batch_size):
                cur_z = self.network.encode(x[i:i+self.batch_size], sample=False)
                if isinstance(cur_z, tuple):  # VAE returns (mu, log_var)
                    cur_z = cur_z[0]
                cur_z = utils.convert_numpy(cur_z)
                z.append(cur_z)
        z = np.concatenate(z, axis=0)
        assert z.shape[0] == n_samples
        return z

    def predict(self, z):
        """ Decode latent factors to recover inputs.
        """
        self.network.eval()
        with torch.no_grad():
            z = torch.tensor(z, dtype=torch.float, device=self.device)
            n_samples = z.shape[0]
            x = []
            for i in range(0, n_samples, self.batch_size):
                cur_x = self.network.decode(z[i:i+self.batch_size])
                cur_x = utils.convert_numpy(cur_x)
                x.append(cur_x)
        x = np.concatenate(x, axis=0)
        assert x.shape[0] == n_samples
        return x

    def _forward(self, batch_data):
        x = torch.tensor(batch_data, dtype=torch.float, device=self.device)
        x_rec, info = self.network.forward(x)
        batch_losses = self.loss_function(x=x, x_rec=x_rec, info=info)
        return batch_losses, x_rec, info

    def fit(self, data, val_data=None):
        """Train. Validation data is optional, only used for logging."""
        # assumed that data is n_samples, n_channels, n_rows, n_columns
        assert np.ndim(data) == 4
        n_samples, input_shape = data.shape[0], data.shape[1:]

        # build the architecture, loss function, optimizer & scheduler
        self.network = nn.get_network(self.architecture_args, [None] + list(input_shape)).to(self.device)
        print(self.network)
        self.loss_function = losses.get_loss_function(self.architecture_args)
        self.optimizer = training.build_optimizer(self.network.parameters(), self.training_args)
        self.scheduler = training.build_scheduler(self.optimizer, self.training_args)

        # tensorboardX writer
        writer = SummaryWriter(self.log_dir)

        for epoch in range(self.epochs):  # outer training loop
            t0 = time.time()
            self.scheduler.step()  # update the learning rate

            # train on the training set
            self.network.train()
            perm = np.random.permutation(n_samples)  # random permutation of data for each epoch
            train_losses = defaultdict(list)
            for i in range(0, n_samples, self.batch_size):  # inner training loop
                batch_data = data[perm[i:i + self.batch_size]]

                # forward pass
                batch_losses, _, _ = self._forward(batch_data)
                batch_total_loss = sum([loss for name, loss in batch_losses.items()])

                # backward pass & update
                self.optimizer.zero_grad()  # PyTorch-specific thing
                batch_total_loss.backward()
                self.optimizer.step()  # update the parameters

                # collect all losses
                if len(batch_losses) > 1:
                    batch_losses['total'] = batch_total_loss
                for k, v in batch_losses.items():
                    train_losses["train_" + k].append(utils.convert_numpy(v))

            for k, v in train_losses.items():
                train_losses[k] = np.mean(v)
                writer.add_scalar('losses/{}'.format(k), np.mean(v), epoch)

            # test on the validation set
            val_losses = defaultdict(list)
            if val_data is not None:
                self.network.eval()  # set testing mode
                for i in range(0, val_data.shape[0], self.batch_size):
                    batch_data = val_data[i:i+self.batch_size]
                    with torch.no_grad():
                        batch_losses, _, _ = self._forward(batch_data)
                        batch_total_loss = sum([loss for name, loss in batch_losses.items()])
                        # collect all losses
                        if len(batch_losses) > 1:
                            batch_losses['total'] = batch_total_loss
                        for k, v in batch_losses.items():
                            val_losses["val_" + k].append(utils.convert_numpy(v))
                for k, v in val_losses.items():
                    val_losses[k] = np.mean(v)
                    writer.add_scalar('losses/{}'.format(k), np.mean(v), epoch)

            if self.verbose:
                t = time.time()
                log_string = 'Epoch: {}/{}'.format(epoch, self.epochs)
                for k, v in list(train_losses.items()) + list(val_losses.items()):
                    log_string += ', {}: {:0.6f}'.format(k, v)
                log_string += ', Time: {:0.1f}s'.format(t - t0)
                print(log_string)

            if (epoch + 1) % self.save_iter == 0:
                utils.save(self,
                           os.path.join(self.log_dir, 'checkpoints', 'epoch{}.mdl'.format(epoch)),
                           self.verbose)

        # save the final version of the network
        utils.save(self, os.path.join(self.log_dir, 'checkpoints', 'final.mdl'), self.verbose)


class IterAE(AE):
    def __init__(self, **kwargs):
        super(IterAE, self).__init__(**kwargs)

    def _forward(self, batch_data):
        # NOTE: assumes the self.network is simple AE
        x = torch.tensor(batch_data, dtype=torch.float, device=self.device)
        z = self.network.encode(x)
        x_rec = self.network.decode(z)
        z_rec = self.network.encode(x_rec)
        x_mse = losses.mse_loss(x, x_rec, None)['mse']
        z_mse = losses.mse_loss(z, z_rec, None)['mse']
        ls = {'x_mse': x_mse, 'z_mse': 1e2 * z_mse}
        info = [z, z_rec]
        return ls, x_rec, info


class IterAESampling(AE):
    def __init__(self, **kwargs):
        super(IterAESampling, self).__init__(**kwargs)

    def _forward(self, batch_data):
        # NOTE: assumes the self.network is simple AE
        if np.random.randint(0, 2) == 1:
            x = torch.tensor(batch_data, dtype=torch.float, device=self.device)
            z = self.network.encode(x)
            x_rec = self.network.decode(z)
            # x_mse = losses.mse_loss(x, x_rec, None)['mse']
            # ls = {'x_mse': x_mse}
            x_ce = losses.cross_entropy_loss(x, x_rec, None)['ce']
            ls = {'x_ce': x_ce}
            info = [z]
            return ls, x_rec, info
        else:
            # z = np.random.uniform(low=-1.0, high=1.0, size=(self.batch_size, self.network.n_hidden))
            z = np.random.normal(0, 1, size=(self.batch_size, self.network.n_hidden))
            z = torch.tensor(z, dtype=torch.float, device=self.device)
            x_rec = self.network.decode(z)
            z_rec = self.network.encode(x_rec)
            z_mse = losses.mse_loss(z, z_rec, None)['mse']
            ls = {'z_mse': 1.0 * z_mse}
            info = [z, z_rec]
        return ls, x_rec, info
