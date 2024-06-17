import time
from os import mkdir

import matplotlib
import numpy as np
import torch
import torch.utils.data
from matplotlib import pyplot as plt


def train_BNN_classification(net, name, batch_size, nb_epochs, trainset, valset, cuda,
                         burn_in, sim_steps, N_saves, resample_its, resample_prior_its,
                         re_burn, flat_ims=False, nb_its_dev=1):
    models_dir = name + '_models'
    results_dir = name + '_results'
    mkdir(models_dir)
    mkdir(results_dir)
    cuda = False

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
## ---------------------------------------------------------------------------------------------------------------------
# net dims
    print('c', '\nNetwork:')
    epoch = 0
    it_count = 0
    ## ---------------------------------------------------------------------------------------------------------------------
    # train
    print('c', '\nTrain:')

    print('  init cost variables:')
    cost_train = np.zeros(nb_epochs)
    err_train = np.zeros(nb_epochs)
    cost_dev = np.zeros(nb_epochs)
    err_dev = np.zeros(nb_epochs)
    best_cost = np.inf
    best_err = np.inf

    tic0 = time.time()
    for i in range(epoch, nb_epochs):
        net.set_mode_train(True)
        tic = time.time()
        nb_samples = 0
        for x, y in trainloader:

            if flat_ims:
                x = x.view(x.shape[0], -1)

            cost_pred, err = net.fit(x, y, burn_in=(i % re_burn < burn_in),
                                     resample_momentum=(it_count % resample_its == 0),
                                     resample_prior=(it_count % resample_prior_its == 0))
            it_count += 1
            err_train[i] += err
            cost_train[i] += cost_pred
            nb_samples += len(x)

        cost_train[i] /= nb_samples
        err_train[i] /= nb_samples
        toc = time.time()

        # ---- print
        print("it %d/%d, Jtr_pred = %f, err = %f, " % (i, nb_epochs, cost_train[i], err_train[i]), end="")
        print('r', '   time: %f seconds\n' % (toc - tic))
        net.update_lr(i)

        # ---- save weights
        if i % re_burn >= burn_in and i % sim_steps == 0:
            net.save_sampled_net(max_samples=N_saves)

        # ---- dev
        if i % nb_its_dev == 0:
            nb_samples = 0
            for j, (x, y) in enumerate(valloader):
                if flat_ims:
                    x = x.view(x.shape[0], -1)

                cost, err, probs = net.eval(x, y)

                cost_dev[i] += cost
                err_dev[i] += err
                nb_samples += len(x)

            cost_dev[i] /= nb_samples
            err_dev[i] /= nb_samples

            print('g', '    Jdev = %f, err = %f\n' % (cost_dev[i], err_dev[i]))
            if err_dev[i] < best_err:
                best_err = err_dev[i]
                print('b', 'best test error')

    toc0 = time.time()
    runtime_per_it = (toc0 - tic0) / float(nb_epochs)
    print('r', '   average time: %f seconds\n' % runtime_per_it)

    ## SAVE WEIGHTS
    net.save_weights(models_dir + '/state_dicts.pkl')

    return cost_train, cost_dev, err_train, err_dev
