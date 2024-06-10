from counterfactual_xai.utils.clue.vae.train import train_VAE
from counterfactual_xai.utils.clue.vae.gaussian_vae import GaussianVAE
from counterfactual_xai.utils.datafeed import DataFeed
from counterfactual_xai.utils.mimic_dataloader import MimiDataLoader
from counterfactual_xai.utils.lsat_dataloader import LsatDataloader  
import torch
import numpy as np

# with hadm_id and mort_hosp
# INPUT_DIMS = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 41]

INPUT_DIMS = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 41]
CSV_PATH = "/vol/fob-vol5/mi22/scholuka/repositorys/counterfactuals/data/cleaned/"

width = 300
depth = 3
latent_dim = 4

x_train, x_test, x_means, x_stds, y_train, y_test, y_means, y_stds, my_data_keys, input_dim_vec = MimiDataLoader(
    INPUT_DIMS, CSV_PATH).get_mimic_dataset()

print('MIMIC', x_train.shape, x_test.shape)
print(input_dim_vec)
print(f"X TRAIN: {x_train}")
print(f"-------------------------------------")
print(f"Y TRAIN: {y_train}")
# input_dim = x_train.shape[1]

for latent_dim in [2, 3, 4, 5, 6, 8, 12, 16]:
    trainset = DataFeed(x_train, x_train, transform=None)
    valset = DataFeed(x_test, x_test, transform=None)

    save_dir = ("/vol/fob-vol5/mi22/scholuka/repositorys/counterfactuals/data/saves/fc_preact_VAE_d2_lsat_models_" + str(latent_dim) + "/")

    batch_size = 128
    nb_epochs = 2500
    lr = 1e-4
    early_stop = 200

    cuda = False

    net = GaussianVAE(input_dim_vec, width, depth, latent_dim, pred_sig=False, lr=lr, cuda=cuda, flatten=False)

    vlb_train, vlb_dev = train_VAE(net, save_dir, batch_size, nb_epochs, trainset, valset,
                                   cuda=cuda, flat_ims=False, train_plot=False, early_stop=early_stop)
