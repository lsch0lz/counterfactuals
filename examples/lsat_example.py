from counterfactual_xai.utils.clue.bnn.gaussian_bnn import GaussianBNN
from counterfactual_xai.utils.clue.gaussian_mlp import GaussianMLP
from counterfactual_xai.utils.datafeed import DataFeed
from counterfactual_xai.utils.lsat_dataloader import LsatDataloader

CSV_PATH = "./../data/"
save_dir = "/Users/lukasscholz/repositorys/CLUE/notebooks/saves/fc_BNN_NEW_ART_lsat_models/state_dicts.pkl"

cuda = False
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
