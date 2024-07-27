from counterfactual_xai.utils.mimic_dataloader import MimiDataLoader
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from counterfactual_xai.utils.lsat_dataloader.lsat_dataloader import LsatDataloader
from counterfactual_xai.utils.datafeed import DataFeed
from counterfactual_xai.utils.clue.gaussian_mlp import GaussianMLP
from counterfactual_xai.utils.clue.bnn.gaussian_bnn import GaussianBNN
from counterfactual_xai.utils.clue.bnn.train_regression import train_BNN_regression

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

y_means = y_train.mean()
y_stds = y_train.std()

trainset = DataFeed(x_train, y_train, transform=None)
valset = DataFeed(x_test, y_test, transform=None)

input_dim = x_train.shape[1]
print(f"Input Dims after shaping: {input_dim}")
print(f"X TRAIN: {x_train[:5]}")
print(f"X TEST: {x_test[:5]}")
print(f"-------------------------------------")
print(f"Y TRAIN: {y_train}")
width = 200
depth = 2
output_dim = 1
model = GaussianMLP(input_dim, width, depth, output_dim, flatten_image=False)

N_train = x_train.shape[0]
batch_size = 512
# nb_epochs = 2200 # We can do less iterations as this method has faster convergence
nb_epochs = 2200
log_interval = 1

lr = 1e-2

## weight saving parameters #######
burn_in = 120  # this is in epochs
sim_steps = 20  # We want less correlated samples -> despite having per minibatch noise we see correlations
N_saves = 100

resample_its = 10
resample_prior_its = 50  # 45 can be choosen to better control overfitting
re_burn = 1e7

cuda = False
net = GaussianBNN(model, N_train, lr=lr, cuda=cuda)

save_dir = "/vol/fob-vol5/mi22/scholuka/repositorys/counterfactuals/data/results/"

cost_train, cost_dev, rms_dev, ll_dev = train_BNN_regression(net, save_dir, batch_size, nb_epochs, trainset, valset,
                                                             cuda,
                                                             burn_in, sim_steps, N_saves, resample_its,
                                                             resample_prior_its,
                                                             re_burn, flat_ims=False, nb_its_dev=10, y_mu=y_means,
                                                             y_std=y_stds)
