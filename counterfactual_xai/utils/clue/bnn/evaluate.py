import torch

from counterfactual_xai.utils.clue.bnn.utils import variable_to_tensor_list
from counterfactual_xai.utils.clue.bnn.gaussian_bnn import GaussianBNN
from counterfactual_xai.utils.clue.gaussian_mlp import GaussianMLP
from counterfactual_xai.utils.datafeed import DataFeed
from counterfactual_xai.utils.mimic_dataloader import MimiDataLoader

def marginal_std(mu, sigma):  # This is for outputs from NN and GMM var estimation
    """Obtain the std of a GMM with isotropic components"""
    # probs (Nsamples, batch_size, classes)
    marg_var = (sigma**2).mean(dim=0) + ((mu ** 2).mean(dim=0) - mu.mean(dim=0) ** 2)
    return torch.sqrt(marg_var)


def evaluate_BNN_net_gauss(net, valloader, y_means, y_stds, samples=0, gmm_sig=False, flat_ims=False):
    mu_vec = []
    sigma_vec = []
    y_vec = []

    for x,y in valloader:
        y, = variable_to_tensor_list(variables=(y,), cuda=net.cuda)
        if flat_ims:
            x = x.view(x.shape[0], -1)
        mu, sig = net.sample_predict(x, num_samples=samples, grad=False)
        mu_vec.append(mu.data.cpu())
        sigma_vec.append(sig.data.cpu())
        y_vec.append(y.data.cpu())

    mu_vec = torch.cat(mu_vec, dim=1)
    sigma_vec = torch.cat(sigma_vec, dim=1)
    y_vec = torch.cat(y_vec, dim=0)

    mu_mean = mu_vec.mean(dim=0)
    sigma_mean = sigma_vec.mean(dim=0)
    marg_sigma = marginal_std(mu_vec, sigma_vec)

    if gmm_sig:
        rms, ll = net.unnormalised_eval(mu_mean, marg_sigma, y_vec, y_mu=y_means, y_std=y_stds)
    else:
        rms, ll = net.unnormalised_eval(mu_mean, sigma_mean, y_vec, y_mu=y_means, y_std=y_stds)

    print('rms', rms, 'll', ll)

    return ll, rms

CSV_PATH = "/Users/lukasscholz/repositorys/studienprojekt/counterfactuals/data/"
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


evaluate_BNN_net_gauss(bnn_gauss, valset, y_means, y_stds)