import numpy as np

import torch
import torch.nn as nn


class LnDistance(nn.Module):
    """If dims is None Compute across all dimensions except first"""

    def __init__(self, n, dim=None):
        super(LnDistance, self).__init__()
        self.n = n
        self.dim = dim

    def forward(self, x, y):
        d = x - y
        if self.dim is None:
            self.dim = list(range(1, len(d.shape)))
        return torch.abs(d).pow(self.n).sum(dim=self.dim).pow(1. / float(self.n))


def calculate_mnist_mean_std_norm(x):
    mean = 0.1307
    std = 0.3081
    x = x - mean
    x = x / std
    return x


def get_num_batches(nb_samples, batch_size, roundup=True):
    if roundup:
        return ((nb_samples + (-nb_samples % batch_size)) / batch_size)  # roundup division
    else:
        return nb_samples / batch_size


def generate_ind_batch(nb_samples, batch_size, random=True, roundup=True):
    if random:
        ind = np.random.permutation(nb_samples)
    else:
        ind = range(int(nb_samples))
    for i in range(int(get_num_batches(nb_samples, batch_size, roundup))):
        yield ind[i * batch_size: (i + 1) * batch_size]
