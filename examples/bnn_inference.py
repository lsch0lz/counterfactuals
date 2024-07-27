import numpy as np
import torch
from sklearn.metrics import mean_squared_error

from counterfactual_xai.utils.clue.bnn.gaussian_bnn import GaussianBNN
from counterfactual_xai.utils.clue.gaussian_mlp import GaussianMLP
from counterfactual_xai.utils.datafeed import DataFeed
from counterfactual_xai.utils.mimic_dataloader import MimiDataLoader

torch.manual_seed(1)

CSV_PATH = "./../data/"
SAVE_DIR = "/Users/lukasscholz/repositorys/studienprojekt/counterfactuals/data/mimic/regression/length_of_stay_regression/2200_epochs/state_dicts.pkl"
CUDA = False
# MORT_ICU
# INPUT_DIMS = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 41]
# LoS Regression
INPUT_DIMS = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 41]
LEARNING_RATE = 1e-2

x_train, x_test, x_means, x_stds, y_train, y_test, y_means, y_stds, DATA_KEYS, input_dims = MimiDataLoader(INPUT_DIMS,
                                                                                                           CSV_PATH).get_mimic_dataset()

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

x_means = x_means.astype(np.float32)
x_stds = x_stds.astype(np.float32)

y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

y_means = y_means.astype(np.float32)
y_stds = y_stds.astype(np.float32)

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
    81.67957059216141,0.2587412587412587,0.4125874125874126,0.0,0.0,0.0,0.0,0.0,0.4125874125874126,0.0,0.2097902097902098,0.0,0.0139860139860139,0.0209790209790209,0.5664335664335665,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0
])

y_pred = []
for x_test in x_test:
    # Normalize the input data
    """ 
    x_input = (sample_input_data - x_means) / x_stds
    x_input = x_input.astype(np.float32)
    
    x_input = torch.tensor(x_input).unsqueeze(0)
    """

    mu, sigma = bnn_gauss.predict(x_test)

    mu_denormalized = mu.detach().numpy() * y_stds + y_means
    sigma_denormalized = sigma.detach().numpy() * y_stds

    y_pred.append(mu_denormalized)

y_pred = np.array([item[0][0] for item in y_pred])
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("MSE: ", mse)
print("RMSE: ", rmse)
