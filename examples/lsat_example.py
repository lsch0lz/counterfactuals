from counterfactual_xai.utils.clue.bnn.gaussian_bnn import GaussianBNN
from counterfactual_xai.utils.clue.gaussian_mlp import GaussianMLP
from counterfactual_xai.utils.clue.vae.gaussian_vae import GaussianVAE
from counterfactual_xai.utils.datafeed import DataFeed
from counterfactual_xai.utils.lsat_dataloader import LsatDataloader

CSV_PATH = "./../data/"
save_dir = "./../data/saves/_models/state_dicts.pkl"

cuda = False
device: str = "cuda" if cuda else "cpu"
learning_rate = 1e-2


INPUT_DIMS = [1, 1, 8, 2]

x_train, x_test, x_means, x_stds, y_train, y_test, y_means, y_stds, DATA_KEYS, input_dims = LsatDataloader(INPUT_DIMS, CSV_PATH).get_lsat_dataset()

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

print('gauss_cat')
lr = 1e-4
VAE = GaussianVAE(input_dims, width, depth, latent_dim, pred_sig=False,
                        lr=lr, cuda=cuda, flatten=False)

VAE.load(filename="./../data/saves/_models/theta_best.dat", device=device)
