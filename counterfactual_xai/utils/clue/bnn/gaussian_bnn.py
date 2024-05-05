import copy
import logging

import numpy as np
import torch
from torch import nn
from torch.backends import cudnn

from counterfactual_xai.utils.base_model import BaseNet
from counterfactual_xai.utils.clue.bnn.stochastic_gradient_hamilton_sampler import StochasticHamiltonMonteCarloSampler
from counterfactual_xai.utils.clue.bnn.utils import variable_to_tensor_list, diagonal_gauss_loglike, gaussian_mixture_model_loglike, \
    get_root_mean_square, \
    save_object, load_object

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class GaussianBNN(BaseNet):
    def __init__(self, model, N_train, lr=1e-2, cuda=True, eps=1e-3, grad_std_mul=20):
        super(GaussianBNN, self).__init__()
        self.lr = lr
        self.model = model
        self.cuda = cuda

        self.N_train = N_train
        self._create_network()
        self._create_optimizer()
        self.schedule = None  # [] #[50,200,400,600]
        self.epoch = 0

        self.grad_buff = []
        self.grad_std_mul = grad_std_mul
        self.max_grad = 1e20
        self.eps = eps

        self.weight_set_samples = []

    def _create_network(self):
        torch.manual_seed(42)
        if self.cuda:
            torch.cuda.manual_seed(42)

        if self.cuda:
            self.model.cuda()
            cudnn.benchmark = True

        print('Total params: %.2fM' % (self.get_nb_parameters() / 1000000.0))

    def _create_optimizer(self):
        self.optimizer = StochasticHamiltonMonteCarloSampler(params=self.model.parameters(), lr=self.lr, base_C=0.05, gauss_sig=0.1)

    def fit(self, x, y, burn_in=False, resample_momentum=False, resample_prior=False):
        self.set_model_mode(train=True)
        x, y = variable_to_tensor_list(variables=(x, y), cuda=self.cuda)

        self.optimizer.zero_grad()
        mu, sigma = self.model(x)
        sigma = sigma.clamp(min=self.eps)
        loss = -diagonal_gauss_loglike(y, mu, sigma).mean(dim=0) * self.N_train

        loss.backward()

        if len(self.grad_buff) > 100:
            self.max_grad = np.mean(self.grad_buff) + self.grad_std_mul * np.std(self.grad_buff)
            self.grad_buff.pop(0)

        self.grad_buff.append(nn.utils.clip_grad_norm_(parameters=self.model.parameters(),
                                                       max_norm=self.max_grad, norm_type=2))
        if self.grad_buff[-1] >= self.max_grad:
            print(self.max_grad, self.grad_buff[-1])
            self.grad_buff.pop()

        self.optimizer.step(burn_in=burn_in, resample_momentum=resample_momentum, resample_prior=resample_prior)

        return loss * x.shape[0] / self.N_train, mu, sigma

    def eval(self, x, y):
        self.set_model_mode(train=False)
        x, y = variable_to_tensor_list(variables=(x, y), cuda=self.cuda)
        mu, sigma = self.model(x)
        sigma = sigma.clamp(min=self.eps)
        loss = -diagonal_gauss_loglike(y, mu, sigma).mean(dim=0) * self.N_train

        return loss * x.shape[0] / self.N_train, mu, sigma

    @staticmethod
    def unnormalised_eval(pred_mu, pred_std, y, y_mu, y_std, gmm=False):
        ll = gaussian_mixture_model_loglike(pred_mu, pred_std, y, y_mu, y_std, gmm=gmm)  # this already computes sum
        if gmm:
            pred_mu = pred_mu.mean(dim=0)
        rms = get_root_mean_square(pred_mu, y, y_mu, y_std)  # this already computes sum
        return rms, ll

    def predict(self, x):
        self.set_model_mode(train=False)
        x, = variable_to_tensor_list(variables=(x,), cuda=self.cuda)
        mu, sigma = self.model(x)
        return mu, sigma

    def save_sampled_net(self, max_samples):

        if len(self.weight_set_samples) >= max_samples:
            self.weight_set_samples.pop(0)

        self.weight_set_samples.append(copy.deepcopy(self.model.state_dict()))

        logger.warning(f"Saving Samples: {len(self.weight_set_samples)} for Max Samples: {max_samples}")
        logger.warning(f"Samples: {self.weight_set_samples}")

        return None

    def sample_predict(self, x, num_samples, grad=False):
        self.set_model_mode(train=False)
        if num_samples == 0:
            num_samples = len(self.weight_set_samples)

        x, = variable_to_tensor_list(variables=(x,), cuda=self.cuda)

        if grad:
            self.optimizer.zero_grad()
            if not x.requires_grad:
                x.requires_grad = True

        mu_vec = x.new(num_samples, x.shape[0], self.model.output_dim)
        std_vec = x.new(num_samples, x.shape[0], self.model.output_dim)

        # iterate over all saved weight configuration samples
        # TODO: mu_vec and std_vec are changed here, prove that the value isnt changed inplace
        for idx, weight_dict in enumerate(self.weight_set_samples):
            if idx == num_samples:
                break
            self.model.load_state_dict(weight_dict)
            mu, std = self.model(x)

            mu_vec[idx] = mu.detach().clone()
            std_vec[idx] = std.detach().clone()

        if grad:
            return mu_vec[:idx], std_vec[:idx]
        else:
            return mu_vec[:idx], std_vec[:idx]

    def get_weight_samples(self, Nsamples=0):
        weight_vec = []

        if Nsamples == 0 or Nsamples > len(self.weight_set_samples):
            Nsamples = len(self.weight_set_samples)

        for idx, state_dict in enumerate(self.weight_set_samples):
            if idx == Nsamples:
                break

            for key in state_dict.keys():
                if 'weight' in key:
                    weight_mtx = state_dict[key].cpu()
                    for weight in weight_mtx.view(-1):
                        weight_vec.append(weight)

        return np.array(weight_vec)

    def save_weights(self, filename):
        save_object(self.weight_set_samples, filename)

    def load_weights(self, filename):
        self.weight_set_samples = load_object(filename)
