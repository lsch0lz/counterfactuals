import torch

from counterfactual_xai.utils.clue.bnn.gaussian_bnn import BNNCategorical
from counterfactual_xai.utils.clue.bnn.mlp import MLP
from counterfactual_xai.utils.datafeed import DataFeed
from counterfactual_xai.utils.mimic_dataloader import MimiDataLoader
from counterfactual_xai.utils.clue.bnn.train_classification import train_BNN_classification

dname = 'compas'

INPUT_DIMS = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 41]
CSV_PATH = "/vol/fob-vol5/mi22/scholuka/repositorys/counterfactuals/data/cleaned/"

x_train, x_test, x_means, x_stds, y_train, y_test, y_means, y_stds, DATA_KEYS, input_dims = MimiDataLoader(INPUT_DIMS,
                                                                                                           CSV_PATH).get_mimic_dataset()
# input_dim_vec = X_dims_to_input_dim_vec(X_dims)

print(x_train.shape)
print(y_test.shape)

trainset = DataFeed(x_train, torch.Tensor(y_train), transform=None)
valset = DataFeed(x_test, torch.Tensor(y_test), transform=None)


input_dim = x_train.shape[1]
width = 200
depth = 2
output_dim = y_train.shape[1]
model = MLP(input_dim, width, depth, output_dim, flatten_image=False)

N_train = len(trainset)
print(N_train)
batch_size = 512#
nb_epochs = 2400 # We can do less iterations as this method has faster convergence

lr = 1e-2

## weight saving parameters #######
burn_in = 120 # this is in epochs
sim_steps = 20 # We want less correlated samples -> despite having per minibatch noise we see correlations
N_saves = 100

resample_its = 10
resample_prior_its = 50
re_burn = 1e7

nb_its_dev = 10

cuda = False
net = BNNCategorical(model, N_train, lr=lr, cuda=cuda)

save_dir = "/vol/fob-vol5/mi22/scholuka/repositorys/counterfactuals/data/results/"

cost_train, cost_dev, err_train, err_dev = train_BNN_classification(net, save_dir, batch_size,
                         nb_epochs, trainset, valset, cuda,
                         burn_in, sim_steps, N_saves, resample_its, resample_prior_its,
                         re_burn, flat_ims=False, nb_its_dev=nb_its_dev)

