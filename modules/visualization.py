""" Visualization routines to use for experiments.

These visualization tools will note save figures. That can be later done by
calling the savefig(fig, path) below. The purpose of this design is to make it
possible to use these tools in both jupyter notebooks and in ordinary scripts.
"""
from modules import utils
import numpy as np
import os

import matplotlib
from matplotlib import pyplot
matplotlib.use('Agg')


def iterate_from_latent(net, z0, n_iterations=20):
    """Repeatedly apply encode(decode(z))."""
    xs = []
    zs = [z0]
    for k in range(n_iterations):
        xs.append(net.predict(zs[-1]))
        zs.append(net.transform(xs[-1]))
    return xs, zs


def iterate_from_input(net, x0, n_iterations=20):
    """Repeatedly apply decode(encode(x))."""
    xs = [x0]
    zs = []
    for k in range(n_iterations):
        zs.append(net.transform(xs[-1]))
        xs.append(net.predict(zs[-1]))
    return xs, zs


def field_flow(net, n_points=40, n_iterations=1, low=-1.0, high=1.0, scale=1.0,
               d1=0, d2=1, plt=None):
    """Visualize the iterative embedding vector field of a model."""
    if plt is None:
        plt = matplotlib.pyplot

    ticks = np.linspace(low, high, n_points)
    x, y = np.meshgrid(ticks, ticks)
    x, y = x.flatten(), y.flatten()
    z = np.random.uniform(low=low, high=high, size=(net.network.n_hidden,))
    z = z.reshape((1, -1)).repeat(x.shape[0], axis=0)
    z[:, d1] = x
    z[:, d2] = y

    _, zs = iterate_from_latent(net, z, n_iterations)
    diff = zs[-1] - zs[0]
    u = diff[:, d1]
    v = diff[:, d2]

    fig, ax = plt.subplots(1)
    alpha = 0.25
    if scale is None:
        alpha = 1.0
    ax.quiver(x, y, u, v, scale=scale, alpha=alpha)
    ax.set_title('Iterative embedding flow')
    return fig, plt


def trajectory_plot(net, n_points=10, n_iterations=10, low=-1.0, high=1.0, d1=0, d2=1, plt=None):
    """ Visualize the iterative embedding vector field of a model."""
    if plt is None:
        plt = matplotlib.pyplot

    fig, ax = plt.subplots(1)
    colors = matplotlib.cm.tab20.colors
    for i in range(n_points):
        z0 = np.random.uniform(low=low, high=high, size=(1, net.network.n_hidden))
        _, zs = iterate_from_latent(net, z0, n_iterations)
        zs = [z[0] for z in zs]
        for j in range(n_iterations):
            ax.arrow(zs[j][d1], zs[j][d2], zs[j+1][d1] - zs[j][d1], zs[j+1][d2] - zs[j][d2],
                     color=colors[i % len(colors)])
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)

    return fig, plt


def reconstruction_plot(net, train_data, val_data, n_samples=3, plt=None):
    """Plots reconstruction examples for training & validation sets."""
    if plt is None:
        plt = pyplot
    samples = np.concatenate([train_data[:n_samples], val_data[:n_samples]], axis=0)
    x_rec = net.predict(net.transform(samples)).reshape(samples.shape)
    fig, ax = plt.subplots(nrows=2 * n_samples, ncols=2, figsize=(2, 2 * n_samples))
    for i in range(2 * n_samples):
        ax[i][0].imshow(samples[i, 0], vmin=0, vmax=1, cmap='gray')
        ax[i][0].set_axis_off()
        ax[i][1].imshow(x_rec[i, 0], vmin=0, vmax=1, cmap='gray')
        ax[i][1].set_axis_off()
    return fig, plt


def manifold_plot(net, low=-1.0, high=+1.0, n_points=20, d1=0, d2=1, plt=None):
    """Plots reconstruction for varying dimensions d1 and d2, while the remaining dimensions are kept fixed."""
    if plt is None:
        plt = pyplot
    image = np.zeros((28 * n_points, 28 * n_points), dtype=np.float32)

    z = np.random.uniform(low=low, high=high, size=(net.network.n_hidden,))
    z1_grid = np.linspace(low, high, n_points)
    z2_grid = np.linspace(low, high, n_points)

    for i, z1 in enumerate(z1_grid):
        for j, z2 in enumerate(z2_grid):
            cur_z = np.copy(z)
            z[d1] = z1
            z[d2] = z2
            cur_z = cur_z.reshape((1, -1))
            x = net.predict(cur_z).reshape((28, 28))
            image[28*i:28*i + 28, 28*j:28*j + 28] = x
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image, vmin=0, vmax=1, cmap='gray')
    ax.axis('off')
    return fig, plt


def latent_scatter(net, data, labels, d1=0, d2=1, plt=None):
    """A scatter plot of latent factors on some 2-d subspace, with points colored according to test labels."""
    if plt is None:
        plt = matplotlib.pyplot
    tab = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
           'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    z = net.transform(data)
    labels = np.array(labels)
    fig, ax = plt.subplots(1)
    legend = []
    for i in np.unique(labels):
        indices = (labels == i)
        ax.scatter(z[indices, d1], z[indices, d2], marker='.', color=tab[i], alpha=0.5, edgecolor='', label=i)
        legend.append(str(i))
    fig.legend(legend)
    ax.set_xlabel("$Z_{}$".format(d1))
    ax.set_ylabel("$Z_{}$".format(d2))
    mu = np.mean(z, axis=0)
    std = np.std(z, axis=0)
    ax.set_xlim(mu[d1] - 3 * std[d1], mu[d1] + 3 * std[d1])
    ax.set_ylim(mu[d2] - 3 * std[d2], mu[d2] + 3 * std[d2])
    ax.set_title('Latent space')
    return fig, plt


def savefig(fig, path):
    utils.make_path(os.path.dirname(path))
    fig.savefig(path)
