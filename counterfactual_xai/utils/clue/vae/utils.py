import torch
from torch.distributions import Normal
from torch.nn.functional import softplus
import torch.nn.functional as F

suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']


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


def torch_one_hot_encoding(y, Nclass):
    if y.is_cuda:
        y = y.type(torch.cuda.LongTensor)
    else:
        y = y.type(torch.LongTensor)
    y_onehot = torch.zeros((y.shape[0], Nclass)).type(y.type())
    y_onehot.scatter_(1, y.unsqueeze(1), 1)
    return y_onehot


def gauss_cat_to_flat(x, input_dim_vec):
    output = []
    for idx, dim in enumerate(input_dim_vec):
        if dim == 1:
            output.append(x[:, idx].unsqueeze(1))
        elif dim > 1:
            oh_vec = torch_one_hot_encoding(x[:, idx], dim).type(x.type())
            output.append(oh_vec)
        else:
            raise ValueError('Error, invalid dimension value')
    return torch.cat(output, dim=1)


def flat_to_gauss_cat(x, input_dim_vec):
    output = []
    cum_dims = 0
    for idx, dims in enumerate(input_dim_vec):
        if dims == 1:
            output.append(x[:, cum_dims].unsqueeze(1))
            cum_dims = cum_dims + 1

        elif dims > 1:
            output.append(x[:, cum_dims:cum_dims + dims].max(dim=1)[1].type(x.type()).unsqueeze(1))
            cum_dims = cum_dims + dims

        else:
            raise ValueError('Error, invalid dimension value')

    return torch.cat(output, dim=1)


def selective_softmax(x, input_dim_vec, grad=False, cat_probs=False, prob_sample=False, eps=1e-20):
    """Applies softmax operation to specified dimensions. Gradient estimator is optional.
    cat_probs returns probability vectors over categorical variables instead of maxing
    if cat_probs is activated with prob sample, a one-hot vector will be sampled (reparametrisable)"""
    output = torch.zeros_like(x)
    cum_dims = 0
    for idx, dim in enumerate(input_dim_vec):
        if dim == 1:
            output[:, cum_dims] = x[:, cum_dims]
            if prob_sample:  # this assumes an rms loss when training
                noise = x.new_zeros(x.shape[0]).normal_(mean=0, std=1)
                output[:, cum_dims] = output[:, cum_dims] + noise
            cum_dims = cum_dims + 1
        elif dim > 1:
            if not cat_probs:
                if not grad:
                    y = x[:, cum_dims:cum_dims + dim].max(dim=1)[1]
                    y_vec = torch_one_hot_encoding(y, dim).type(x.type())
                    output[:, cum_dims:cum_dims + dim] = y_vec
                else:
                    x_cat = x[:, cum_dims:cum_dims + dim]
                    probs = F.softmax(x_cat, dim=1)
                    y_hard = x[:, cum_dims:cum_dims + dim].max(dim=1)[1]
                    y_oh = torch_one_hot_encoding(y_hard, dim).type(x.type())
                    output[:, cum_dims:cum_dims + dim] = (y_oh - probs).detach() + probs
            else:
                x_cat = x[:, cum_dims:cum_dims + dim]
                probs = F.softmax(x_cat, dim=1)

                if prob_sample:  # we are going to use gumbel trick here
                    log_probs = torch.log(probs)
                    u = log_probs.new(log_probs.shape).uniform_(0, 1)
                    g = -torch.log(-torch.log(u + eps) + eps)
                    cat_samples = (log_probs + g).max(dim=1)[1]
                    hard_samples = torch_one_hot_encoding(cat_samples, dim).type(x.type())
                    output[:, cum_dims:cum_dims + dim] = hard_samples
                else:
                    output[:, cum_dims:cum_dims + dim] = probs

            cum_dims = cum_dims + dim
        else:
            raise ValueError('Error, invalid dimension value')
    return output


def humansize(nbytes):
    i = 0
    while nbytes >= 1024 and i < len(suffixes) - 1:
        nbytes /= 1024.
        i = i + 1
    f = ('%.2f' % nbytes)
    return '%s%s' % (f, suffixes[i])
