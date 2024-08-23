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


class SkipConnection(nn.Module):
    """
    Skip-connection over the sequence of layers in the constructor.
    The module passes input data sequentially through these layers
    and then adds original data to the result.
    """

    def __init__(self, *args):
        super(SkipConnection, self).__init__()
        self.inner_net = nn.Sequential(*args)

    def forward(self, input):
        return input + self.inner_net(input)


def preact_leaky_MLPBlock(width):
    return SkipConnection(
        nn.LeakyReLU(),
        nn.BatchNorm1d(num_features=width),
        nn.Linear(width, width),
    )


class MLP_preact_prior_net(nn.Module):
    def __init__(self, input_dim, width, depth, latent_dim):
        super(MLP_preact_prior_net, self).__init__()
        # input layer
        proposal_layers = [nn.Linear(input_dim * 2, width)]
        # body
        for i in range(depth - 1):
            proposal_layers.append(preact_leaky_MLPBlock(width))
        # output layer
        proposal_layers.extend([nn.LeakyReLU(), nn.BatchNorm1d(num_features=width), nn.Linear(width, latent_dim * 2)])

        self.block = nn.Sequential(*proposal_layers)

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
                cum_dims = cum_dims + 1
            elif dims > 1:
                if x.shape[1] == y.shape[1]:
                    raise Exception('Input and target seem to be in flat format. Need integer cat targets.')
                if y.is_cuda:
                    tget = y[:, idx].type(torch.cuda.LongTensor)
                else:
                    tget = y[:, idx].type(torch.LongTensor)

                log_prob_vec.append(-self.ce(x[:, cum_dims:cum_dims + dims], tget).unsqueeze(1))
                cum_dims = cum_dims + dims
            else:
                raise ValueError('Error, invalid dimension value')

        log_prob_vec = torch.cat(log_prob_vec, dim=1)

        if self.reduction == 'none':
            return log_prob_vec
        elif self.reduction == 'sum':
            return log_prob_vec.sum()
        elif self.reduction == 'average':
            return log_prob_vec.mean()


from torch.nn.functional import softplus
from torch.distributions import Normal


def normal_parse_params(params, min_sigma=1e-3):
    """
    Take a Tensor (e. g. neural network output) and return
    torch.distributions.Normal distribution.
    This Normal distribution is component-wise independent,
    and its dimensionality depends on the input shape.
    First half of channels is mean of the distribution,
    the softplus of the second half is std (sigma), so there is
    no restrictions on the input tensor.
    min_sigma is the minimal value of sigma. I. e. if the above
    softplus is less than min_sigma, then sigma is clipped
    from below with value min_sigma. This regularization
    is required for the numerical stability and may be considered
    as a neural network architecture choice without any change
    to the probabilistic model.
    """
    n = params.shape[0]
    d = params.shape[1]
    mu = params[:, :d // 2]
    sigma_params = params[:, d // 2:]
    sigma = softplus(sigma_params)
    sigma = sigma.clamp(min=min_sigma)
    distr = Normal(mu, sigma)
    return distr


class GaussianLoglike(nn.Module):
    """
    Compute reconstruction log probability of groundtruth given
    a tensor of Gaussian distribution parameters and a mask.
    Gaussian distribution parameters are output of a neural network
    without any restrictions, the minimal sigma value is clipped
    from below to min_sigma (default: 1e-2) in order not to overfit
    network on some exact pixels.
    The first half of channels corresponds to mean, the second half
    corresponds to std. See normal_parse_parameters for more info.
    This layer doesn't work with NaNs in the data, it is used for
    inpainting. Roughly speaking, this loss is similar to L2 loss.
    Returns a vector of log probabilities for each object of the batch.
    """

    def __init__(self, min_sigma=1e-2):
        super(GaussianLoglike, self).__init__()
        self.min_sigma = min_sigma

    def forward(self, distr_params, groundtruth, mask=None):
        distr = normal_parse_params(distr_params, self.min_sigma)
        if mask is not None:
            log_probs = distr.log_prob(groundtruth) * mask
        else:
            log_probs = distr.log_prob(groundtruth)
        return log_probs.view(groundtruth.shape[0], -1).sum(-1)
