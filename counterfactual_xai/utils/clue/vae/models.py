import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from counterfactual_xai.utils.clue.vae.layers import preact_leaky_mlp_block


class MLPPreactRecognitionNetwork(nn.Module):
    def __init__(self, input_dim, width, depth, latent_dim):
        super(MLPPreactRecognitionNetwork, self).__init__()
        # input layer
        proposal_layers = [nn.Linear(input_dim, width)]
        # body
        for i in range(depth - 1):
            proposal_layers.append(preact_leaky_mlp_block(width))
        # output layer
        proposal_layers.extend(
            [nn.LeakyReLU(), nn.BatchNorm1d(num_features=width), nn.Linear(width, latent_dim * 2)])

        self.block = nn.Sequential(*proposal_layers)

    def forward(self, x):
        return self.block(x)


class MLPPreactGeneratorNetwork(nn.Module):
    def __init__(self, input_dim, width, depth, latent_dim):
        super(MLPPreactGeneratorNetwork, self).__init__()
        # input layer
        generative_layers = [nn.Linear(latent_dim, width), nn.LeakyReLU(), nn.BatchNorm1d(num_features=width)]
        # body
        for i in range(depth - 1):
            generative_layers.append(preact_leaky_mlp_block(width))
        # output layer
        generative_layers.extend([nn.Linear(width, input_dim), ])
        self.block = nn.Sequential(*generative_layers)

    def forward(self, x):
        return self.block(x)


class RMSCatLoglike(nn.Module):
    def __init__(self, input_dim_vec, reduction='none'):
        super(RMSCatLoglike, self).__init__()
        self.reduction = reduction
        self.input_dim_vec = input_dim_vec
        self.mse = MSELoss(reduction='none')  # takes(input, target)
        self.ce = CrossEntropyLoss(reduction='none')

    def forward(self, x, y):
        log_prob_vec = []
        cum_dims = 0
        for idx, dims in enumerate(self.input_dim_vec):
            if dims == 1:
                # Gaussian_case
                log_prob_vec.append(-self.mse(x[:, cum_dims], y[:, idx]).unsqueeze(1))
                cum_dims += 1
            elif dims > 1:
                if x.shape[1] == y.shape[1]:
                    raise Exception('Input and target seem to be in flat format. Need integer cat targets.')
                if y.is_cuda:
                    tget = y[:, idx].type(torch.cuda.LongTensor)
                else:
                    tget = y[:, idx].type(torch.LongTensor)

                log_prob_vec.append(-self.ce(x[:, cum_dims:cum_dims + dims], tget).unsqueeze(1))
                cum_dims += dims
            else:
                raise ValueError('Error, invalid dimension value')

        log_prob_vec = torch.cat(log_prob_vec, dim=1)

        if self.reduction == 'none':
            return log_prob_vec
        elif self.reduction == 'sum':
            return log_prob_vec.sum()
        elif self.reduction == 'average':
            return log_prob_vec.mean()
