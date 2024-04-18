import numpy as np
import torch
import torch.utils.data


def decompose_std_gauss(mu, sigma, sum_dims=True):
    # probs (Nsamples, batch_size, output_sims)
    aleatoric_var = (sigma ** 2).mean(dim=0)
    epistemic_var = ((mu ** 2).mean(dim=0) - mu.mean(dim=0) ** 2)
    total_var = aleatoric_var + epistemic_var
    if sum_dims:
        aleatoric_var = aleatoric_var.sum(dim=1)
        epistemic_var = epistemic_var.sum(dim=1)
        total_var = total_var.sum(dim=1)
    return total_var.sqrt(), aleatoric_var.sqrt(), epistemic_var.sqrt()


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
        zz = VAE.recongnition(x).loc.data.cpu().numpy()
        # Note that naming is wrong and this is actually std instead of entropy
        if prob_BNN:
            mu_vec, std_vec = BNN.sample_predict(x, 0, False)
            total_entropy, aleatoric_entropy, epistemic_entropy = decompose_std_gauss(mu_vec, std_vec)
        else:
            mu, std = BNN.predict(x, grad=False)
            total_entropy = std
            aleatoric_entropy = std
            epistemic_entropy = std*0

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
