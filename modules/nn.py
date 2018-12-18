import numpy as np
from torch import nn
import torch


""" Some functions for building basic blocks """


def infer_shape(layers, input_shape):
    input_shape = [x for x in input_shape]
    if input_shape[0] is None:
        input_shape[0] = 4  # should be more than 1, otherwise batch norm will not work
    x = torch.tensor(np.random.normal(size=input_shape), dtype=torch.float, device='cpu')
    for layer in layers:
        x = layer(x)
    output_shape = list(x.shape)
    output_shape[0] = None
    return output_shape


def add_activation(layers, activation):
    if activation == 'relu':
        layers.append(nn.ReLU())
    if activation == 'sigmoid':
        layers.append(nn.Sigmoid())
    if activation == 'tanh':
        layers.append(nn.Tanh())
    if activation == 'softplus':
        layers.append(nn.Softplus())
    if activation == 'linear':
        pass
    return layers


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self._shape = tuple([-1, ] + list(shape))

    def forward(self, x):
        return x.view(self._shape)


def parse_args(architecture_args, input_shape):
    net = nn.ModuleList()
    for cur_layer in architecture_args:
        layer_type = cur_layer['type']
        print(infer_shape(net, input_shape))

        if layer_type == 'fc':
            dim = cur_layer['dim']
            prev_shape = infer_shape(net, input_shape)
            assert len(prev_shape) == 2
            net.append(nn.Linear(prev_shape[1], dim))
            if cur_layer.get('batch_norm', False):
                net.append(nn.BatchNorm1d(dim))
            add_activation(net, cur_layer.get('activation', 'linear'))

        if layer_type == 'flatten':
            net.append(Flatten())

        if layer_type == 'reshape':
            net.append(Reshape(cur_layer['shape']))

        if layer_type == 'conv':
            prev_shape = infer_shape(net, input_shape)
            assert len(prev_shape) == 4
            net.append(nn.Conv2d(
                in_channels=prev_shape[1],
                out_channels=cur_layer['filters'],
                kernel_size=cur_layer['kernel_size'],
                stride=cur_layer['stride'],
                padding=cur_layer.get('padding', 0)
            ))
            if cur_layer.get('batch_norm', False):
                net.append(torch.nn.BatchNorm2d(
                    num_features=cur_layer['filters']))
            add_activation(net, cur_layer.get('activation', 'linear'))

        if layer_type == 'deconv':
            prev_shape = infer_shape(net, input_shape)
            assert len(prev_shape) == 4
            net.append(nn.ConvTranspose2d(
                in_channels=prev_shape[1],
                out_channels=cur_layer['filters'],
                kernel_size=cur_layer['kernel_size'],
                stride=cur_layer['stride'],
                padding=cur_layer.get('padding', 0),
                output_padding=cur_layer.get('output_padding', 0)
            ))
            if cur_layer.get('batch_norm', False):
                net.append(torch.nn.BatchNorm2d(
                    num_features=cur_layer['filters']))
            add_activation(net, cur_layer.get('activation', 'linear'))

    output_shape = infer_shape(net, input_shape)
    print(output_shape)
    return net, output_shape


""" Standard auto-encoder & VAE """


class AutoencoderBase(nn.Module):
    """
    General class for autoencoders. Can handle VAEs too.

    NOTE: encode/decode functions have **kwargs to allow passing
          additional arguments, such as whether to sample or not.
    """
    def __init__(self):
        super(AutoencoderBase, self).__init__()

    def encode(self, x, **kwargs):
        out = x
        for layer in self._encoder:
            out = layer(out)
        return out

    def decode(self, z, **kwargs):
        out = z
        for layer in self._decoder:
            out = layer(out)
        return out

    def forward(self, x):
        """Forward returns the reconstruction plus additional info we need for losses or measures."""
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec, [z]


class AE(AutoencoderBase):
    def __init__(self, architecture_args, input_shape):
        super(AE, self).__init__()
        self.input_shape = input_shape

        # build encoder
        self._encoder, hidden_shape = parse_args(architecture_args['encoder'], input_shape)
        assert len(hidden_shape) == 2
        self.n_hidden = hidden_shape[1]

        # build decoder
        self._decoder, self.output_shape = parse_args(architecture_args['decoder'], hidden_shape)


class VAE(AutoencoderBase):
    def __init__(self, architecture_args, input_shape):
        super(VAE, self).__init__()
        self.input_shape = input_shape

        # build encoder
        self._encoder, encoder_shape = parse_args(architecture_args['encoder'], input_shape)

        # build hidden & sampling
        self.n_hidden = architecture_args['n_hidden']
        assert len(encoder_shape) == 2
        self.get_mu = nn.Linear(encoder_shape[1], self.n_hidden)
        self.get_log_var = nn.Linear(encoder_shape[1], self.n_hidden)
        hidden_shape = (None, self.n_hidden)

        # build decoder
        self._decoder, self.output_shape = parse_args(architecture_args['decoder'], hidden_shape)

    @staticmethod
    def sample(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def encode(self, x, sample=False, **kwargs):
        out = x
        for layer in self._encoder:
            out = layer(out)
        mu = self.get_mu(out)
        log_var = self.get_log_var(out)
        if sample:
            return VAE.sample(mu, log_var)
        return mu, log_var

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = VAE.sample(mu, log_var)
        x_rec = self.decode(z)
        return x_rec, [z, mu, log_var]


def get_network(architecture_args, input_shape):
    t = architecture_args.get('type', 'ae')
    if t == 'ae':
        return AE(architecture_args, input_shape)
    if t == 'vae':
        return VAE(architecture_args, input_shape)
    return None
