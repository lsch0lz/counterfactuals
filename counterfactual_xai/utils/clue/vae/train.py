import time

import numpy as np
import torch
import torch.utils.data
from torchvision.utils import make_grid
from numpy.random import normal

from counterfactual_xai.utils.clue.vae.utils import torch_one_hot_encoding, humansize


def train_VAE(net, name, batch_size, nb_epochs, trainset, valset, cuda, flat_ims=False,
              train_plot=False, Nclass=None, early_stop=None, script_mode=False):
    models_dir = name + '_models'
    results_dir = name + '_results'

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

    epoch = 0

    vlb_train = np.zeros(nb_epochs)
    vlb_dev = np.zeros(nb_epochs)
    best_vlb = -np.inf
    best_vlb_train = -np.inf
    best_epoch = 0

    nb_its_dev = 1

    tic0 = time.time()
    for i in range(epoch, nb_epochs):
        net.set_model_mode(True)

        nb_samples = 0
        for x, y in trainloader:

            if flat_ims:
                x = x.view(x.shape[0], -1)
            if Nclass is not None:
                y_oh = torch_one_hot_encoding(y, Nclass).type(x.type())
                x = torch.cat([x, y_oh], 1)

            cost, _ = net.fit(x)

            vlb_train[i] += cost * len(x)
            nb_samples += len(x)

        vlb_train[i] /= nb_samples

        net.update_lr(i)

        if vlb_train[i] > best_vlb_train:
            best_vlb_train = vlb_train[i]

        # ---- dev
        if i % nb_its_dev == 0:
            nb_samples = 0
            for j, (x, y) in enumerate(valloader):

                if flat_ims:
                    x = x.view(x.shape[0], -1)
                if Nclass is not None:
                    y_oh = torch_one_hot_encoding(y, Nclass).type(x.type())
                    x = torch.cat([x, y_oh], 1)

                cost, _ = net.eval(x)

                vlb_dev[i] += cost * len(x)
                nb_samples += len(x)

            vlb_dev[i] /= nb_samples

            if train_plot:
                zz = net.recongnition(x).sample()
                o = net.regenerate(zz)
                try:
                    o = o.cpu()
                except:
                    o = o.loc.cpu()
                if len(x.shape) == 2:
                    side = int(np.sqrt(x.shape[1]))
                    x = x.view(-1, 1, side, side).data
                    o = o.view(-1, 1, side, side).data

                # save_image(torch.cat([x[:8], o[:8]]), results_dir + '/rec_%d.png' % i, nrow=8)
                import matplotlib.pyplot as plt
                plt.figure()
                dd = make_grid(torch.cat([x[:10], o[:10]]), nrow=10).numpy()
                plt.imshow(np.transpose(dd, (1, 2, 0)), interpolation='nearest')
                if script_mode:
                    plt.savefig(results_dir + '/rec%d.png' % i)
                else:
                    plt.show()

                z_sample = normal(loc=0.0, scale=1.0, size=(36, net.latent_dim))
                x_rec = net.regenerate(z_sample)
                try:
                    x_rec = x_rec.cpu()
                except:
                    x_rec = x_rec.loc.cpu()
                if len(x_rec.shape) == 2:
                    side = int(np.sqrt(x_rec.shape[1]))
                    x_rec = x_rec.view(-1, 1, side, side)
                plt.figure()
                dd = make_grid(x_rec, nrow=6).numpy()
                plt.imshow(np.transpose(dd, (1, 2, 0)), interpolation='nearest')
                if script_mode:
                    plt.savefig(results_dir + '/sample%d.png' % i)
                else:
                    plt.show()

        if vlb_dev[i] > best_vlb:
            best_vlb = vlb_dev[i]
            best_epoch = i
            net.save(models_dir + '/theta_best.dat')

        if early_stop is not None and (i - best_epoch) > early_stop:
            break

    net.save(models_dir + '/theta_last.dat')

    nb_parameters = net.get_nb_parameters()
    best_cost_dev = best_vlb
    best_cost_train = best_vlb_train

    print('  best_vlb_dev: %f' % best_cost_dev)
    print('  best_vlb_train: %f' % best_cost_train)
    print('  nb_parameters: %d (%s)\n' % (nb_parameters, humansize(nb_parameters)))

    ## ---------------------------------------------------------------------------------------------------------------------
    # fig cost vs its
    if not train_plot:
        import matplotlib
        matplotlib.use('agg')
    import matplotlib.pyplot as plt
    if train_plot:
        plt.figure()
        plt.plot(np.clip(vlb_train, -1000, 1000), 'r')
        plt.plot(np.clip(vlb_dev[::nb_its_dev], -1000, 1000), 'b')
        plt.legend(['cost_train', 'cost_dev'])
        plt.ylabel('vlb')
        plt.xlabel('it')
        plt.grid(True)
        plt.savefig(results_dir + '/train_cost.png')
        if train_plot:
            plt.show()
    return vlb_train, vlb_dev
