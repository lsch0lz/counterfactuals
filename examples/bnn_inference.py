import numpy as np
import torch

from counterfactual_xai.utils.clue.bnn.gaussian_bnn import GaussianBNN
from counterfactual_xai.utils.clue.gaussian_mlp import GaussianMLP
from counterfactual_xai.utils.datafeed import DataFeed
from counterfactual_xai.utils.mimic_dataloader import MimiDataLoader

torch.manual_seed(1)

CSV_PATH = "./../data/"
SAVE_DIR = "/Users/lukasscholz/repositorys/studienprojekt/counterfactuals/data/mimic/results/_model_mort_icu_cleaned/state_dicts.pkl"
CUDA = False
INPUT_DIMS = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 41]
LEARNING_RATE = 1e-2

x_train, x_test, x_means, x_stds, y_train, y_test, y_means, y_stds, DATA_KEYS, input_dims = MimiDataLoader(INPUT_DIMS,
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

sample_input_data = np.array([
    85.9525519178862, 74, 0.96, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0266666666666666, 0.0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
])

# Normalize the input data
x_input = (sample_input_data - x_means) / x_stds
x_input = x_input.astype(np.float32)

x_input = torch.tensor(x_input).unsqueeze(0)

mu, sigma = bnn_gauss.predict(x_input)

mu_denormalized = mu.detach().numpy() * y_stds + y_means
sigma_denormalized = sigma.detach().numpy() * y_stds

abs_mu_denormalized = np.abs(mu_denormalized)

print("Predicted mean:", mu_denormalized)
print("Predicted standard deviation:", sigma_denormalized)
print("Predicted mean (absolute value):", abs_mu_denormalized)
