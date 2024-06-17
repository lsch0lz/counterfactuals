from typing import List

import numpy as np
import torch
import pandas as pd

from sklearn.metrics import mean_absolute_error

from counterfactual_xai.utils.clue.bnn.gaussian_bnn import GaussianBNN
from counterfactual_xai.utils.clue.gaussian_mlp import GaussianMLP
from counterfactual_xai.utils.datafeed import DataFeed
from counterfactual_xai.utils.mimic_dataloader import MimiDataLoader

torch.manual_seed(1)

CSV_PATH = "./../data/"
SAVE_DIR = "/Users/lukasscholz/repositorys/studienprojekt/counterfactuals/data/mimic/regression/length_of_stay_regression/adjusted_mean_as_return_type/state_dicts.pkl"
CUDA = False
# MORT_ICU
# INPUT_DIMS = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 41]
# LoS Regression
INPUT_DIMS = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 41]
LEARNING_RATE = 1e-2

x_train, x_test, x_means_adjusted, x_stds_adjusted, y_train, y_test, y_means, y_stds, DATA_KEYS, input_dims = MimiDataLoader(INPUT_DIMS,
                                                                                                                             CSV_PATH).get_mimic_dataset()

trainset = DataFeed(x_train, y_train, transform=None)
valset = DataFeed(x_test, y_test, transform=None)

input_dim = x_train.shape[1]
width = 200
depth = 2
output_dim = y_train.shape[1]

mlp_gauss = GaussianMLP(input_dim, width, depth, output_dim, flatten_image=False)
N_train = x_train.shape[0]

bnn_gauss = GaussianBNN(mlp_gauss, N_train, lr=LEARNING_RATE, cuda=CUDA)
bnn_gauss.load_weights(SAVE_DIR)

df_test = pd.read_csv("/Users/lukasscholz/repositorys/studienprojekt/counterfactuals/data/test_length_of_stay_prediction.csv")

predicted_max_hours: List[float] = []
real_max_hours: List[float] = []

for idx, row in df_test.iterrows():
    train_sample = row.to_list()
    gt_max_hour = train_sample[1]
    del train_sample[1]

    # Normalize the input data
    train_sample = np.array(train_sample)
    x_input = (train_sample - x_means_adjusted) / x_stds_adjusted
    x_input = x_input.astype(np.float32)

    x_input = torch.tensor(x_input).unsqueeze(0)

    mu, sigma = bnn_gauss.predict(x_input)

    mu_denormalized = mu.detach().numpy() * y_stds + y_means

    print(f"Predicted: {mu_denormalized.item()}")
    print(f"Truth: {gt_max_hour}")

    predicted_max_hours.append(mu_denormalized.item())
    real_max_hours.append(gt_max_hour)


score = mean_absolute_error(real_max_hours, predicted_max_hours)
print("The Mean Absolute Error of our Model is {}".format(round(score, 2)))

score = np.sqrt(mean_absolute_error(real_max_hours, predicted_max_hours))
print("The Mean Absolute Error of our Model is {}".format(round(score, 2)))