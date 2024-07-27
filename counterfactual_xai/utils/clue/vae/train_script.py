from counterfactual_xai.utils.clue.vae.train import train_VAE
from counterfactual_xai.utils.clue.vae.gaussian_vae import GaussianVAE
from counterfactual_xai.utils.datafeed import DataFeed
from counterfactual_xai.utils.mimic_dataloader import MimiDataLoader
from counterfactual_xai.utils.lsat_dataloader import LsatDataloader  
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CSV_PATH = "/vol/fob-vol5/mi22/scholuka/repositorys/counterfactuals/data/"

width = 300
depth = 3
latent_dim = 4

input_dim_vec = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 29]
df_clean = pd.read_csv("/vol/fob-vol5/mi22/scholuka/repositorys/counterfactuals/data/los_prediction.csv")

LOS = df_clean['LOS'].values
# Prediction Features
features = df_clean.drop(columns=['LOS'])

x_train, x_test, y_train, y_test = train_test_split(features,
                                                    LOS,
                                                    test_size=.20,
                                                    random_state=0)

x_train = x_train.to_numpy()
x_test = x_test.to_numpy()

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

# x_train = torch.tensor(x_train)
# x_test = torch.tensor(x_test)

y_means = y_train.mean()
y_stds = y_train.std()

print('MIMIC', x_train.shape, x_test.shape)
print(input_dim_vec)
print(f"X TRAIN: {x_train}")
print(f"-------------------------------------")
print(f"Y TRAIN: {y_train}")

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
