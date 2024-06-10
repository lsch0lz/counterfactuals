from __future__ import print_function
from __future__ import division

from os import mkdir

import torch
import torch.utils.data
import numpy as np


def train_BNN_regression(model, model_name, batch_size, epochs, trainset, valset, cuda,
                         burn_in, sim_steps, N_saves, resample_its, resample_prior_its,
                         re_burn, flat_ims=False, nb_its_dev=1, y_mu=0, y_std=1):
    models_dir = model_name + '_models'
    results_dir = model_name + '_results'
    mkdir(models_dir)
    mkdir(results_dir)

    epoch = 0
    it_count = 0

    if cuda:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                                  num_workers=3)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                                num_workers=3)
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=False,
                                                  num_workers=3)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=False,
                                                num_workers=3)

    print('Init cost variables:')
    cost_train = np.zeros(epochs)
    cost_dev = np.zeros(epochs)
    ll_dev = np.zeros(epochs)
    rms_dev = np.zeros(epochs)
    best_cost = np.inf

    for i in range(epoch, epochs):
        model.set_model_mode(True)
        num_samples = 0

        for x, y in trainloader:
            if flat_ims:
                x = x.view(x.shape[0], -1)

            cost_pred, _, _ = model.fit(x, y, burn_in=(i % re_burn < burn_in),
                                        resample_momentum=(it_count % resample_its == 0),
                                        resample_prior=(it_count % resample_prior_its == 0))
            it_count += 1
            cost_train[i] += cost_pred
            num_samples += len(x)

        cost_train[i] /= num_samples

        print("it %d/%d, Jtr_pred = %f, " % (i, epochs, cost_train[i]), end="")
        model.update_lr(i)

        if i % re_burn >= burn_in and i % sim_steps == 0:
            model.save_sampled_net(max_samples=N_saves)

        if i % nb_its_dev == 0:
            num_samples = 0
            mu_vec = []
            sigma_vec = []
            y_vec = []
            for j, (x, y) in enumerate(valloader):
                if flat_ims:
                    x = x.view(x.shape[0], -1)

                cost, mu, sigma = model.eval(x, y)
                mu_vec.append(mu.data.cpu())
                sigma_vec.append(sigma.data.cpu())
                y_vec.append(y.data.cpu())

                cost_dev[i] += cost
                num_samples += len(x)

            mu_vec = torch.cat(mu_vec)
            sigma_vec = torch.cat(sigma_vec)
            y_vec = torch.cat(y_vec)
            rms, ll = model.unnormalised_eval(mu_vec, sigma_vec, y_vec, y_mu, y_std)
            ll_dev[i] = ll
            rms_dev[i] = rms
            cost_dev[i] /= num_samples
            if cost_dev[i] < best_cost:
                best_cost = cost_dev[i]

    model.save_weights(models_dir + '/state_dicts.pkl')

    return cost_train, cost_dev, rms_dev, ll_dev
