import numpy as np

import torch
import torch.utils.data
from matplotlib import pyplot as plt

from counterfactual_xai.methods.clue import decompose_std_gauss, decompose_entropy_cat


def latent_project_gauss(BNN, VAE, dset, batch_size=1024, cuda=True, prob_BNN=True):
    if cuda:
        loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                             num_workers=3)
    else:
        loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False, pin_memory=False,
                                             num_workers=3)
    z_train = []
    y_train = []
    x_train = []
    tr_aleatoric_vec = []
    tr_epistemic_vec = []

    for j, (x, y_l) in enumerate(loader):
        zz = VAE.recongnition(x).loc.detach().cpu().numpy()
        # Note that naming is wrong and this is actually std instead of entropy
        if prob_BNN:
            mu_vec, std_vec = BNN.sample_predict(x, 0, False)
            total_entropy, aleatoric_entropy, epistemic_entropy = decompose_std_gauss(mu_vec, std_vec)
        else:
            mu, std = BNN.predict(x, grad=False)
            total_entropy = std
            aleatoric_entropy = std
            epistemic_entropy = std * 0

        tr_epistemic_vec.append(epistemic_entropy)
        tr_aleatoric_vec.append(aleatoric_entropy)

        z_train.append(zz)
        y_train.append(y_l.numpy())
        x_train.append(x.numpy())

    tr_aleatoric_vec = torch.cat(tr_aleatoric_vec).detach().cpu().numpy()
    tr_epistemic_vec = torch.cat(tr_epistemic_vec).detach().cpu().numpy()
    z_train = np.concatenate(z_train)
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    return tr_aleatoric_vec, tr_epistemic_vec, z_train, x_train, y_train


def latent_project_cat(BNN, VAE, dset, batch_size=1024, cuda=True, prob_BNN=True):
    if cuda:
        loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                             num_workers=3)
    else:
        loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False, pin_memory=False,
                                             num_workers=3)
    z_train = []
    y_train = []
    x_train = []
    tr_aleatoric_vec = []
    tr_epistemic_vec = []

    for j, (x, y_l) in enumerate(loader):
        zz = VAE.recongnition(x).loc.data.cpu().numpy()

        # print(x.shape)
        if prob_BNN:
            probs = BNN.sample_predict(x, 0, False)
            total_entropy, aleatoric_entropy, epistemic_entropy = decompose_entropy_cat(probs)
        else:
            probs = BNN.predict(x, grad=False)
            total_entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1, keepdim=False)
            aleatoric_entropy = total_entropy
            epistemic_entropy = total_entropy * 0

        tr_epistemic_vec.append(epistemic_entropy.data)
        tr_aleatoric_vec.append(aleatoric_entropy.data)

        z_train.append(zz)
        y_train.append(y_l.numpy())
        x_train.append(x.numpy())

    tr_aleatoric_vec = torch.cat(tr_aleatoric_vec).cpu().numpy()
    tr_epistemic_vec = torch.cat(tr_epistemic_vec).cpu().numpy()
    z_train = np.concatenate(z_train)
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    return tr_aleatoric_vec, tr_epistemic_vec, z_train, x_train, y_train


def gen_bar_plot(labels, data, title=None, xlabel=None, ylabel=None, probs=False, hor=False, save_file=None,
                 max_fields=40, fs=7, verbose=False, sort=False, dpi=40, neg_color=True, ax=None, c=None):
    if c is None:
        c = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    c1 = c[0]
    c2 = c[1]  # ???

    add_tit = ''
    if data.shape[0] > max_fields or sort:
        if verbose:
            print(title + ' Demasiados campos de datos, mostrando %d mas grandes' % max_fields)
        add_tit = ' (top %d)' % max_fields
        abs_data = np.abs(data)
        sort_idx = np.flipud(np.argsort(abs_data))[:max_fields]
        labels = labels[sort_idx]
        data = data[sort_idx]

    if ax == None:
        plt.figure(dpi=dpi, figsize=(20, 12))  # Increased figure size
        ax = plt.gca()

    fst = 15

    if neg_color:
        c = np.array([c1] * data.shape[0])
        c[data < 0] = c2
    else:
        c = c1

    if hor:
        ax.barh(labels, data, 0.8, color=c)
        ax.invert_yaxis()
        ax.set_yticklabels(labels, fontsize=fs)  # Set y-tick label size
    else:
        ax.bar(labels, data, color=c)
        ax.set_xticklabels(labels, fontsize=fs, rotation=45, ha='right')  # Rotate x-tick labels

    ax.xaxis.grid(alpha=0.35)
    ax.yaxis.grid(alpha=0.35)

    if title is not None:
        plt.title(title + add_tit)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if probs and hor:
        ax.set_xlim((0, 1))
    elif probs and not hor:
        ax.set_ylim((0, 1))

    ax.title.set_fontsize(fst)
    for item in ([ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fs)
        item.set_weight('normal')
    ax.legend(prop={'size': fs, 'weight': 'normal'}, frameon=False)

    ax.autoscale(enable=True, axis='x', tight=True)
    ax.autoscale(enable=True, axis='y', tight=True)
    #     plt.tight_layout()

    if save_file is not None:
        plt.savefig(save_file, bbox_inches='tight')

    return ax
