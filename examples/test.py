import torch

import numpy as np

from counterfactual_xai.utils.clue.bnn.gaussian_bnn import GaussianBNN
from counterfactual_xai.utils.clue.gaussian_mlp import GaussianMLP
from counterfactual_xai.utils.clue.vae.gaussian_vae import GaussianVAE
from counterfactual_xai.utils.datafeed import DataFeed
from counterfactual_xai.utils.lsat_dataloader import LsatDataloader
from counterfactual_xai.methods.interpretation import latent_project_gauss
from counterfactual_xai.methods.utils import LnDistance
from counterfactual_xai.methods.clue import decompose_entropy_cat, decompose_std_gauss, CLUE

torch.autograd.set_detect_anomaly(True)


def main():
    CSV_PATH = "./../data/"
    save_dir = "./../data/saves/_models/state_dicts.pkl"

    cuda = False
    device: str = "cuda" if cuda else "cpu"
    learning_rate = 1e-2

    INPUT_DIMS = [1, 1, 8, 2]

    x_train, x_test, x_means, x_stds, y_train, y_test, y_means, y_stds, DATA_KEYS, input_dims = LsatDataloader(INPUT_DIMS,
                                                                                                               CSV_PATH).get_lsat_dataset()

    trainset = DataFeed(x_train, y_train, transform=None)
    valset = DataFeed(x_test, y_test, transform=None)

    input_dim = x_train.shape[1]
    width = 200
    depth = 2
    output_dim = y_train.shape[1]

    mlp_gauss = GaussianMLP(input_dim, width, depth, output_dim, flatten_image=False)
    print(mlp_gauss.eval())

    N_train = x_train.shape[0]

    bnn_gauss = GaussianBNN(mlp_gauss, N_train, lr=learning_rate, cuda=cuda)
    bnn_gauss.load_weights(save_dir)

    width = 300
    depth = 3
    latent_dim = 4

    lr = 1e-4
    VAE = GaussianVAE(input_dims, width, depth, latent_dim, pred_sig=False,
                      lr=lr, cuda=cuda, flatten=False)

    VAE.load(filename="./../data/saves/_models/theta_best.dat", device=device)

    tr_aleatoric_vec, tr_epistemic_vec, z_train, x_train, y_train = latent_project_gauss(bnn_gauss, VAE, dset=trainset,
                                                                                         batch_size=2048, cuda=cuda)

    tr_uncertainty_vec = tr_aleatoric_vec + tr_epistemic_vec

    te_aleatoric_vec, te_epistemic_vec, z_test, x_test, y_test = latent_project_gauss(bnn_gauss, VAE, dset=valset,
                                                                                      batch_size=2048, cuda=cuda)

    te_uncertainty_vec = (te_aleatoric_vec ** 2 + te_epistemic_vec ** 2) ** (1.0 / 2)

    uncertainty_idxs_sorted = np.flipud(np.argsort(te_uncertainty_vec))
    aleatoric_idxs_sorted = np.flipud(np.argsort(te_aleatoric_vec))
    epistemic_idxs_sorted = np.flipud(np.argsort(te_epistemic_vec))

    dname = "lsat"
    var_names = {"lsat": ['LSAT', 'UGPA', 'race', 'sex']}
    var_names_flat = {
        "lsat": ['LSAT', 'UGPA', 'amerind', 'mexican', 'other', 'black', 'asian', 'puerto', 'hisp', 'white', 'female',
                 'male']}

    var_N = 5


    bins = [-5, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 5]
    center_bins = ((np.array([0] + bins) + np.array(bins + [0])) / 2)[1:]
    center_bins[-1] = bins[-1]

    bin_idx = np.digitize(x_train[:, var_N], bins, right=False)
    bin_means = []
    bin_stds = []
    aleatoric_mean = []
    aleatoric_stds = []
    epistemic_mean = []
    epistemic_stds = []

    for n_bin, bin_start in enumerate(bins):
        y_select = y_train[bin_idx == n_bin]
        aleatoric_select = tr_aleatoric_vec[bin_idx == n_bin]
        epistemic_select = tr_epistemic_vec[bin_idx == n_bin]
        if len(y_select) == 0:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)
            aleatoric_mean.append(np.nan)
            aleatoric_stds.append(np.nan)
            epistemic_mean.append(np.nan)
            epistemic_stds.append(np.nan)
        else:
            bin_means.append(y_select.mean())
            bin_stds.append(y_select.std())
            aleatoric_mean.append(aleatoric_select.mean())
            aleatoric_stds.append(aleatoric_select.std())
            epistemic_mean.append(epistemic_select.mean())
            epistemic_stds.append(epistemic_select.std())

    # plt.figure(dpi=80)

    use_index = uncertainty_idxs_sorted

    Nbatch = 512
    z_init_batch = z_test[use_index[:Nbatch]]
    x_init_batch = x_test[use_index[:Nbatch]]
    y_init_batch = y_test[use_index[:Nbatch]]

    from counterfactual_xai.methods.clue import CLUE

    torch.cuda.empty_cache()

    dist = LnDistance(n=1, dim=1)
    x_dim = x_init_batch.reshape(x_init_batch.shape[0], -1).shape[1]

    aleatoric_weight = 0
    epistemic_weight = 0
    uncertainty_weight = 1

    # Both weights are set to 1?
    distance_weight = 1.5 / x_dim
    prediction_similarity_weight = 0

    mu_vec, std_vec = bnn_gauss.sample_predict(x_init_batch, 0, grad=False)
    o_uncert, o_aleatoric, o_epistemic = decompose_std_gauss(mu_vec, std_vec)
    desired_preds = mu_vec.mean(dim=0).detach().cpu().numpy()

    CLUE_explainer = CLUE(VAE, bnn_gauss, x_init_batch, uncertainty_weight=uncertainty_weight,
                          aleatoric_weight=aleatoric_weight, epistemic_weight=epistemic_weight,
                          prior_weight=0, distance_weight=distance_weight,
                          latent_L2_weight=0, prediction_similarity_weight=prediction_similarity_weight,
                          lr=1e-2, desired_preds=None, cond_mask=None, distance_metric=dist,
                          z_init=z_init_batch, norm_MNIST=False,
                          flatten_BNN=False, regression=True, cuda=False)

    # clue_instance.optimizer = SGD(self.trainable_params, lr=lr, momentum=0.5, nesterov=True)
    z_vec, x_vec, uncertainty_vec, epistemic_vec, aleatoric_vec, cost_vec, dist_vec = CLUE_explainer.optimise(
        min_steps=3, max_steps=35,
        n_early_stop=3)


if __name__ == "__main__":
    main()
