import _pickle
import pickle
from typing import Tuple

import numpy as np
import torch

from torch.autograd import Variable


def variable_to_tensor_list(variables: Tuple, cuda=True, volatile=False):
    tensor_variables = []
    for variable in variables:
        if isinstance(variable, np.ndarray):
            variable = torch.from_numpy(variable).type(torch.FloatTensor)
        if not variable.is_cuda and cuda:
            variable = variable.cuda()
        if not isinstance(variable, Variable):
            variable = Variable(variable, volatile=volatile)
        tensor_variables.append(variable)
    return tensor_variables


def diagonal_gauss_loglike(x, mu, sigma):
    # note that we can just treat each dim as isotropic and then do sum
    cte_term = -0.5 * np.log(2 * np.pi)
    det_sig_term = -torch.log(torch.tensor(sigma))
    inner = (x - mu) / sigma
    dist_term = -(0.5) * (inner ** 2)
    log_px = (cte_term + det_sig_term + dist_term).sum(dim=1, keepdim=False)

    return log_px


def gaussian_mixutre_model_likelihood(x, mu_vec, sigma_vec):
    weight_factor = np.log(mu_vec.shape[0])
    loglike_terms = []
    for i in range(mu_vec.shape[0]):
        loglike_terms.append(diagonal_gauss_loglike(x, mu_vec[i], sigma_vec[i]))
    loglike_terms = torch.cat(loglike_terms, dim=0)

    out = torch.logsumexp(loglike_terms, dim=0) - weight_factor

    return out


def gaussian_mixture_model_loglike(mu, sigma, y, y_means, y_stds, gmm=False):
    mu_un = mu.detach().numpy() * y_stds + y_means
    y_un = y * y_stds + y_means
    sigma_un = sigma.detach().numpy() * y_stds
    if gmm:
        ll = gaussian_mixutre_model_likelihood(y_un, mu_un, sigma_un)
    else:
        ll = diagonal_gauss_loglike(y_un, mu_un, sigma_un)

    return ll.mean(dim=0)


def get_root_mean_square(mu, y, y_means, y_stds):
    x_un = mu * y_stds + y_means
    y_un = y * y_stds + y_means
    return torch.sqrt(((x_un - y_un) ** 2).sum() / y.shape[0])


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as input:
        try:
            return pickle.load(input)
        except UnicodeDecodeError:
            try:
                return pickle.load(input, fix_imports=True, encoding='bytes')
            except _pickle.UnpicklingError:
                return pickle.load(input, fix_imports=True, encoding="latin1")
