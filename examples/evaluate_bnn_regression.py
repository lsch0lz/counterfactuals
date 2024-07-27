from typing import List

import numpy as np
import torch
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from counterfactual_xai.utils.clue.bnn.gaussian_bnn import GaussianBNN
from counterfactual_xai.utils.clue.gaussian_mlp import GaussianMLP
from counterfactual_xai.utils.datafeed import DataFeed
from counterfactual_xai.utils.mimic_dataloader import MimiDataLoader

torch.manual_seed(1)

SAVE_DIR = "/Users/lukasscholz/repositorys/studienprojekt/counterfactuals/data/mimic/regression/length_of_stay_regression/cleaned_dataset/state_dicts.pkl"
LEARNING_RATE = 1e-2


df_clean = pd.read_csv("/Users/lukasscholz/repositorys/studienprojekt/counterfactuals/data/mimic/los_prediction.csv")

LOS = df_clean['LOS'].values
# Prediction Features
features = df_clean.drop(columns=['LOS'])

x_train, x_test, y_train, y_test = train_test_split(features,
                                                    LOS,
                                                    test_size=.20,
                                                    random_state=0)

y_stds = y_train.std()
y_means = y_train.mean()
x_train = x_train.to_numpy()
x_test = x_test.to_numpy()

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

trainset = DataFeed(x_train, y_train, transform=None)
valset = DataFeed(x_test, y_test, transform=None)

input_dim = x_train.shape[1]
width = 200
depth = 2
output_dim = 1

mlp_gauss = GaussianMLP(input_dim, width, depth, output_dim, flatten_image=False)
N_train = x_train.shape[0]

bnn_gauss = GaussianBNN(mlp_gauss, N_train, lr=LEARNING_RATE, cuda=False)
bnn_gauss.load_weights(SAVE_DIR)

def calculate_mse_rmse(x_test, y_test):
    y_pred = []
    for val in x_test:
        mu, sigma = bnn_gauss.predict(val)
        mu_denormalized = mu.detach().numpy() * y_stds + y_means
        y_pred.append(mu_denormalized)

    y_pred = np.array([item[0][0] for item in y_pred])
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    return mse, rmse


mse, rmse = calculate_mse_rmse(x_test, y_test)

print(f"MSE: {mse}")

print(f"RMSE: {rmse}")