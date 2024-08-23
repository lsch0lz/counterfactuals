import numpy as np
import torch
from torch import nn
from torch.backends import cudnn
from torch.distributions import kl_divergence
from torch.distributions.normal import Normal
from torch.nn import MSELoss

from counterfactual_xai.utils.base_model import BaseNet
from counterfactual_xai.utils.clue.bnn.utils import variable_to_tensor_list
from counterfactual_xai.utils.clue.vae.models import MLPPreactRecognitionNetwork, MLPPreactGeneratorNetwork, RMSCatLoglike, MLP_preact_prior_net, \
    GaussianLoglike
from counterfactual_xai.utils.clue.vae.radam import RAdam
from counterfactual_xai.utils.clue.vae.utils import gauss_cat_to_flat, flat_to_gauss_cat, normal_parse_params, selective_softmax, \
    gauss_cat_to_flat_mask


class VAEGaussianCat(nn.Module):
    def __init__(self, input_dim_vec, width, depth, latent_dim, pred_sig=False):
        super(VAEGaussianCat, self).__init__()

        input_dim = 0
        self.input_dim_vec = input_dim_vec
        for e in input_dim_vec:
            input_dim = input_dim + e

        self.encoder = MLPPreactRecognitionNetwork(input_dim, width, depth, latent_dim)
        if pred_sig:
            raise NotImplementedError()
        else:
            self.decoder = MLPPreactGeneratorNetwork(input_dim, width, depth, latent_dim)
            self.rec_loglike = RMSCatLoglike(self.input_dim_vec, reduction='none')
        self.pred_sig = pred_sig

    def encode(self, x):
        """Works with flattened representATION"""
        approx_post_params = self.encoder(x)
        approx_post = normal_parse_params(approx_post_params, 1e-3)
        return approx_post

    def decode(self, z_sample):
        """Works with flattened representATION"""
        rec_params = self.decoder(z_sample)
        return rec_params

    def vlb(self, prior, approx_post, x, rec_params):
        """Works with flattened representATION"""
        if self.pred_sig:
            pass
        else:
            rec = self.rec_loglike(rec_params, x).view(x.shape[0], -1).sum(-1)
        kl = kl_divergence(approx_post, prior).view(x.shape[0], -1).sum(-1)
        return rec - kl

    def iwlb(self, prior, approx_post, x, K=50):
        estimates = []
        for i in range(K):
            latent = approx_post.rsample()
            rec_params = self.decode(latent)
            if self.pred_sig:
                pass
            else:
                rec_loglike = self.rec_loglike(rec_params, x).view(x.shape[0], -1).sum(-1)

            prior_log_prob = prior.log_prob(latent)
            prior_log_prob = prior_log_prob.view(x.shape[0], -1)
            prior_log_prob = prior_log_prob.sum(-1)

            proposal_log_prob = approx_post.log_prob(latent)
            proposal_log_prob = proposal_log_prob.view(x.shape[0], -1)
            proposal_log_prob = proposal_log_prob.sum(-1)

            estimate = rec_loglike + prior_log_prob - proposal_log_prob
            estimates.append(estimate[:, None])

        return torch.logsumexp(torch.cat(estimates, 1), 1) - np.log(K)


class GaussianVAE(BaseNet):
    def  __init__(self, input_dim_vec, width, depth, latent_dim, pred_sig=False, lr=1e-3, cuda=True, flatten=True):
        super(GaussianVAE, self).__init__()

        self.cuda = cuda
        self.input_dim = 0
        self.input_dim_vec = input_dim_vec
        for e in self.input_dim_vec:
            self.input_dim = self.input_dim + e
        self.flatten = flatten
        if not self.flatten:
            pass

        self.width = width
        self.depth = depth
        self.latent_dim = latent_dim
        self.lr = lr
        self.pred_sig = pred_sig

        self.create_net()
        self.create_opt()
        self.epoch = 0
        self.schedule = None

        if self.cuda:
            self.prior = self.prior = Normal(loc=torch.zeros(latent_dim).cuda(), scale=torch.ones(latent_dim).cuda())
        else:
            self.prior = Normal(loc=torch.zeros(latent_dim), scale=torch.ones(latent_dim))
        self.vlb_scale = 1 / len(self.input_dim_vec)  # scale for dimensions of input so we can use same LR always

    def create_net(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        self.model = VAEGaussianCat(self.input_dim_vec, self.width, self.depth, self.latent_dim, self.pred_sig)
        if self.cuda:
            self.model = self.model.cuda()
            cudnn.benchmark = True
        print('    Total params: %.2fM' % (self.get_nb_parameters() / 1000000.0))

    def create_opt(self):
        self.optimizer = RAdam(self.model.parameters(), lr=self.lr)

    def fit(self, x):
        self.set_mode_train(train=True)

        if self.flatten:
            x_flat = gauss_cat_to_flat(x, self.input_dim_vec)
        else:
            x_flat = x
            x = flat_to_gauss_cat(x, self.input_dim_vec)

        x, x_flat = variable_to_tensor_list(variables=(x, x_flat), cuda=self.cuda)
        self.optimizer.zero_grad()

        approx_post = self.model.encode(x_flat)
        z_sample = approx_post.rsample()
        rec_params = self.model.decode(z_sample)

        vlb = self.model.vlb(self.prior, approx_post, x, rec_params)
        loss = (- vlb * self.vlb_scale).mean()

        loss.backward()
        self.optimizer.step()

        return vlb.mean().item(), rec_params

    def eval(self, x, sample=False):
        self.set_mode_train(train=False)

        if self.flatten:
            x_flat = gauss_cat_to_flat(x, self.input_dim_vec)
        else:
            x_flat = x
            x = flat_to_gauss_cat(x, self.input_dim_vec)

        x, x_flat = variable_to_tensor_list(variables=(x, x_flat), cuda=self.cuda)
        approx_post = self.model.encode(x_flat)
        if sample:
            z_sample = approx_post.sample()
        else:
            z_sample = approx_post.loc
        rec_params = self.model.decode(z_sample)

        vlb = self.model.vlb(self.prior, approx_post, x, rec_params)

        return vlb.mean().item(), rec_params

    def eval_iw(self, x, k=50):
        self.set_mode_train(train=False)
        if self.flatten:
            x_flat = gauss_cat_to_flat(x, self.input_dim_vec)
        else:
            x_flat = x
            x = flat_to_gauss_cat(x, self.input_dim_vec)

        x, x_flat = variable_to_tensor_list(variables=(x, x_flat), cuda=self.cuda)

        approx_post = self.model.recognition_encode(x)

        iw_lb = self.model.iwlb(self.prior, approx_post, x, k)
        return iw_lb.mean().item()

    def recongnition(self, x, grad=False, flatten=None):
        if flatten is None:
            flatten = self.flatten
        if flatten and grad:
            raise Exception('flatten and grad options are not compatible')
        self.set_mode_train(train=False)
        if flatten:
            x = gauss_cat_to_flat(x, self.input_dim_vec)
        if grad:
            if not x.requires_grad:
                x.requires_grad = True
        else:
            x, = variable_to_tensor_list(variables=(x,), volatile=True, cuda=self.cuda)
        approx_post = self.model.encode(x)
        return approx_post

    def regenerate(self, z, grad=False, unflatten=False):
        if unflatten and grad:
            raise Exception('flatten and grad options are not compatible')
        self.set_mode_train(train=False)
        if grad:
            if not z.requires_grad:
                z.requires_grad = True
        else:
            z, = variable_to_tensor_list(variables=(z,), volatile=True, cuda=self.cuda)
        out = self.model.decode(z)

        if unflatten:
            out = flat_to_gauss_cat(out, self.input_dim_vec)
        else:
            out = selective_softmax(out, self.input_dim_vec, grad=grad)

        if self.pred_sig:
            raise Exception('Not implemented')
        else:
            return out

class VAEAC_gauss_cat(nn.Module):
    def __init__(self, input_dim_vec, width, depth, latent_dim, pred_sig=True):
        super(VAEAC_gauss_cat, self).__init__()

        self.input_dim = 0
        self.input_dim_vec = input_dim_vec
        for e in self.input_dim_vec:
            self.input_dim += e

        self.latent_dim = latent_dim
        self.recognition_net = MLPPreactRecognitionNetwork(self.input_dim, width, depth, latent_dim)
        self.prior_net = MLP_preact_prior_net(self.input_dim, width, depth, latent_dim)
        if pred_sig:
            raise Exception('Not yet implemented')
        else:
            self.generator_net = MLPPreactGeneratorNetwork(self.input_dim, width, depth, latent_dim)
            self.rec_loglike = RMSCatLoglike(self.input_dim_vec, reduction='none')
        self.pred_sig = pred_sig
        self.sigma_mu = 1e4
        self.sigma_sigma = 1e-4


    @staticmethod
    def apply_mask(x, mask):
        """Positive bits in mask are set to 0 in x (observed)"""
        observed = x.clone()  # torch.tensor(x)
        observed[mask.bool()] = 0
        return observed

    def recognition_encode(self, x):
        """Works with flattened representATION"""
        approx_post_params = self.recognition_net(x)
        approx_post = normal_parse_params(approx_post_params, 1e-3)
        return approx_post

    def prior_encode(self, x, mask):
        """Works with flattened representATION"""
        x = self.apply_mask(x, mask)
        x = torch.cat([x, mask], 1)
        prior_params = self.prior_net(x)
        prior = normal_parse_params(prior_params, 1e-3)
        return prior

    def decode(self, z_sample):
        rec_params = self.generator_net(z_sample)
        return rec_params

    def reg_cost(self, prior):
        num_objects = prior.mean.shape[0]
        mu = prior.mean.view(num_objects, -1)
        sigma = prior.scale.view(num_objects, -1)
        mu_regularizer = -(mu ** 2).sum(-1) / 2 / (self.sigma_mu ** 2)
        sigma_regularizer = (sigma.log() - sigma).sum(-1) * self.sigma_sigma
        return mu_regularizer + sigma_regularizer

    def vlb(self, prior, approx_post, x, rec_params):
        if self.pred_sig:
            raise Exception('Not yet imeplemented')
        else:
            rec = self.rec_loglike(rec_params, x).view(x.shape[0], -1).sum(-1)
        prior_regularization = self.reg_cost(prior).view(x.shape[0], -1).sum(-1)
        kl = kl_divergence(approx_post, prior).view(x.shape[0], -1).sum(-1)
        return rec - kl + prior_regularization

    def iwlb(self, prior, approx_post, x, K=50):
        estimates = []
        for i in range(K):
            latent = approx_post.rsample()
            rec_params = self.decode(latent)
            if self.pred_sig:
                raise Exception('Not yet imeplemented')
            else:
                rec_loglike = self.rec_loglike(rec_params, x).view(x.shape[0], -1).sum(-1)

            prior_log_prob = prior.log_prob(latent)
            prior_log_prob = prior_log_prob.view(x.shape[0], -1)
            prior_log_prob = prior_log_prob.sum(-1)

            proposal_log_prob = approx_post.log_prob(latent)
            proposal_log_prob = proposal_log_prob.view(x.shape[0], -1)
            proposal_log_prob = proposal_log_prob.sum(-1)

            estimate = rec_loglike + prior_log_prob - proposal_log_prob
            estimates.append(estimate[:, None])

        return torch.logsumexp(torch.cat(estimates, 1), 1) - np.log(K)


class VAEACGaussCatNet(BaseNet):
    def __init__(self, input_dim_vec, width, depth, latent_dim, pred_sig=False, lr=1e-3, cuda=True, flatten=True):
        super(VAEACGaussCatNet, self).__init__()
        print('y', 'VAE_gauss_net')

        self.cuda = cuda

        self.input_dim = 0
        self.input_dim_vec = input_dim_vec
        for e in self.input_dim_vec:
            self.input_dim += e
        self.flatten = flatten
        if not self.flatten:
            pass
            # raise Exception('Error calculation not supported without flattening')

        self.width = width
        self.depth = depth
        self.latent_dim = latent_dim
        self.lr = lr
        self.pred_sig = pred_sig

        self.create_net()
        self.create_opt()
        self.epoch = 0
        self.schedule = None

        self.vlb_scale = 1 / len(self.input_dim_vec)  # scale for dimensions of input so we can use same LR always

    def create_net(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        self.model = VAEAC_gauss_cat(self.input_dim_vec, self.width, self.depth, self.latent_dim, self.pred_sig)
        if self.cuda:
            self.model = self.model.cuda()
            cudnn.benchmark = True
        print('    Total params: %.2fM' % (self.get_nb_parameters() / 1000000.0))

    def create_opt(self):
        self.optimizer = RAdam(self.model.parameters(), lr=self.lr)  # torch.optim.Adam

    def fit(self, x, mask):
        self.set_mode_train(train=True)

        if self.flatten:
            mask = gauss_cat_to_flat_mask(mask, self.input_dim_vec)
            x_flat = gauss_cat_to_flat(x, self.input_dim_vec)
        else:
            x_flat = x
            x = flat_to_gauss_cat(x, self.input_dim_vec)

        x_flat, x, mask = variable_to_tensor_list(variables=(x_flat, x, mask), cuda=self.cuda)
        self.optimizer.zero_grad()

        prior = self.model.prior_encode(x_flat, mask)
        approx_post = self.model.recognition_encode(x_flat)
        z_sample = approx_post.rsample()
        rec_params = self.model.decode(z_sample)

        vlb = self.model.vlb(prior, approx_post, x, rec_params)
        loss = (- vlb * self.vlb_scale).mean()

        loss.backward()
        self.optimizer.step()

        if self.pred_sig:
            rec_return = normal_parse_params(rec_params, 1e-3)
        else:
            rec_return = rec_params
        return vlb.mean().item(), rec_return

    def eval(self, x, mask, sample=False):
        self.set_mode_train(train=False)

        if self.flatten:
            mask = gauss_cat_to_flat_mask(mask, self.input_dim_vec)
            x_flat = gauss_cat_to_flat(x, self.input_dim_vec)
        else:
            x_flat = x
            x = flat_to_gauss_cat(x, self.input_dim_vec)

        x_flat, x, mask = variable_to_tensor_list(variables=(x_flat, x, mask), cuda=self.cuda)
        prior = self.model.prior_encode(x_flat, mask)

        approx_post = self.model.recognition_encode(x_flat)
        if sample:
            z_sample = approx_post.sample()
        else:
            z_sample = approx_post.loc
        rec_params = self.model.decode(z_sample)

        vlb = self.model.vlb(prior, approx_post, x, rec_params)

        if self.pred_sig:
            rec_return = normal_parse_params(rec_params, 1e-3)
        else:
            rec_return = rec_params
        return vlb.mean().item(), rec_return

    def eval_iw(self, x, mask, k=50):
        self.set_mode_train(train=False)

        if self.flatten:
            mask = gauss_cat_to_flat_mask(mask, self.input_dim_vec)
            x_flat = gauss_cat_to_flat(x, self.input_dim_vec)
        else:
            x_flat = x
            x = flat_to_gauss_cat(x, self.input_dim_vec)


        x_flat, x, mask = variable_to_tensor_list(variables=(x_flat, x, mask), cuda=self.cuda)

        prior = self.model.prior_encode(x, mask)
        approx_post = self.model.recognition_encode(x_flat)

        iw_lb = self.model.iwlb(prior, approx_post, x, k)
        return iw_lb.mean().item()

    def get_prior(self, x, mask, flatten=None):
        self.set_mode_train(train=False)
        if flatten is None:
            flatten = self.flatten
        if flatten:
            mask = gauss_cat_to_flat_mask(mask, self.input_dim_vec)
            x = gauss_cat_to_flat(x, self.input_dim_vec)

        x, mask = variable_to_tensor_list(variables=(x, mask), cuda=self.cuda)
        prior = self.model.prior_encode(x, mask)
        return prior

    def get_post(self, x, flatten=None):
        self.set_mode_train(train=False)
        if flatten is None:
            flatten = self.flatten
        if flatten:
            x = gauss_cat_to_flat(x, self.input_dim_vec)

        x, = variable_to_tensor_list(variables=(x,), cuda=self.cuda)
        approx_post = self.model.recognition_encode(x)
        return approx_post

    def inpaint(self, x, mask, Nsample=1, z_mean=False, flatten=False, unflatten=False, cat_probs=False):
        self.set_mode_train(train=False)

        if flatten:
            mask = gauss_cat_to_flat_mask(mask, self.input_dim_vec)
            x = gauss_cat_to_flat(x, self.input_dim_vec)

        x, mask = variable_to_tensor_list(variables=(x, mask), cuda=self.cuda)
        prior = self.model.prior_encode(x, mask)
        out = []
        for i in range(Nsample):
            if z_mean:
                z_sample = prior.loc.data
            else:
                z_sample = prior.sample()
            rec_params = self.model.decode(z_sample)
            out.append(rec_params.data)
        out = torch.stack(out, dim=0)

        if self.pred_sig:
            raise Exception('Not yet implemented')
        else:
            dim0 = out.shape[0]
            dim1 = out.shape[1]
            out = out.view(dim0*dim1, -1)
            if unflatten:
                out = flat_to_gauss_cat(out, self.input_dim_vec)
            else:
                out = selective_softmax(out, self.input_dim_vec, grad=False, cat_probs=cat_probs)
            out = out.view(dim0, dim1, -1)
            return out.data

    def regenerate(self, z, grad=False, unflatten=False):
        self.set_mode_train(train=False)
        if unflatten and grad:
            raise Exception('flatten and grad options are not compatible')
        if grad:
            if not z.requires_grad:
                z.requires_grad = True
        else:
            z, = variable_to_tensor_list(variables=(z,), volatile=True, cuda=self.cuda)
        out = self.model.decode(z)
        if self.pred_sig:
            raise Exception('Not yet implemented')
        else:
            if unflatten:
                out = flat_to_gauss_cat(out, self.input_dim_vec)
            else:
                out = selective_softmax(out, self.input_dim_vec, grad=grad)
            return out.data

class VAE_gauss(nn.Module):
    def __init__(self, input_dim, width, depth, latent_dim, pred_sig=True):
        super(VAE_gauss, self).__init__()

        self.encoder = MLPPreactRecognitionNetwork(input_dim, width, depth, latent_dim)
        if pred_sig:
            self.decoder = MLPPreactGeneratorNetwork(2*input_dim, width, depth, latent_dim)
            self.rec_loglike = GaussianLoglike(min_sigma=1e-2)
        else:
            self.decoder = MLPPreactGeneratorNetwork(input_dim, width, depth, latent_dim)
            self.m_rec_loglike = MSELoss(reduction='none')
        self.pred_sig = pred_sig

    def encode(self, x):
        approx_post_params = self.encoder(x)
        approx_post = normal_parse_params(approx_post_params, 1e-3)
        return approx_post

    def decode(self, z_sample):
        rec_params = self.decoder(z_sample)
        return rec_params

    def vlb(self, prior, approx_post, x, rec_params):
        if self.pred_sig:
            rec = self.rec_loglike(rec_params, x).view(x.shape[0], -1).sum(-1)
        else:
            rec = -self.m_rec_loglike(rec_params, x).view(x.shape[0], -1).sum(-1)
        kl = kl_divergence(approx_post, prior).view(x.shape[0], -1).sum(-1)
        return rec - kl

    def iwlb(self, prior, approx_post, x, K=50):
        estimates = []
        for i in range(K):
            latent = approx_post.rsample()
            rec_params = self.decode(latent)
            if self.pred_sig:
                rec_loglike = self.rec_loglike(rec_params, x).view(x.shape[0], -1).sum(-1)
            else:
                rec_loglike = -self.m_rec_loglike(rec_params, x).view(x.shape[0], -1).sum(-1)

            prior_log_prob = prior.log_prob(latent)
            prior_log_prob = prior_log_prob.view(x.shape[0], -1)
            prior_log_prob = prior_log_prob.sum(-1)

            proposal_log_prob = approx_post.log_prob(latent)
            proposal_log_prob = proposal_log_prob.view(x.shape[0], -1)
            proposal_log_prob = proposal_log_prob.sum(-1)

            estimate = rec_loglike + prior_log_prob - proposal_log_prob
            estimates.append(estimate[:, None])

        return torch.logsumexp(torch.cat(estimates, 1), 1) - np.log(K)


class under_VAEAC(BaseNet):
    def __init__(self, base_VAE, width, depth, latent_dim, lr=1e-3, cuda=True):
        super(under_VAEAC, self).__init__()
        print('y', 'VAE_gauss_net')

        self.base_VAEAC = base_VAE
        self.pred_sig = False

        self.cuda = cuda

        self.input_dim = self.base_VAEAC.latent_dim
        self.width = width
        self.depth = depth
        self.latent_dim = latent_dim
        self.lr = lr

        self.create_net()
        self.create_opt()
        self.epoch = 0
        self.schedule = None

        if self.cuda:
            self.prior = self.prior = Normal(loc=torch.zeros(latent_dim).cuda(), scale=torch.ones(latent_dim).cuda())
        else:
            self.prior = Normal(loc=torch.zeros(latent_dim), scale=torch.ones(latent_dim))
        self.vlb_scale = 1 / self.input_dim  # scale for dimensions of input so we can use same LR always

    def create_net(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        self.model = VAE_gauss(self.input_dim, self.width, self.depth, self.latent_dim, self.pred_sig)
        if self.cuda:
            self.model = self.model.cuda()
            cudnn.benchmark = True
        print('    Total params: %.2fM' % (self.get_nb_parameters() / 1000000.0))

    def create_opt(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, x):
        self.set_mode_train(train=True)

        x, = variable_to_tensor_list(variables=(x, ), cuda=self.cuda)
        self.optimizer.zero_grad()

        z_sample = self.base_VAEAC.recognition_encode(x).sample()
        approx_post = self.model.encode(z_sample)
        u_sample = approx_post.rsample()
        rec_params = self.model.decode(u_sample)

        vlb = self.model.vlb(self.prior, approx_post, z_sample, rec_params)
        loss = (- vlb * self.vlb_scale).mean()

        loss.backward()
        self.optimizer.step()

        return vlb.mean().item(), rec_params

    def eval(self, x):
        self.set_mode_train(train=False)

        x, = variable_to_tensor_list(variables=(x, ), cuda=self.cuda)
        z_sample = self.base_VAEAC.recognition_encode(x).sample()
        approx_post = self.model.encode(z_sample)
        u_sample = approx_post.rsample()
        rec_params = self.model.decode(u_sample)

        vlb = self.model.vlb(self.prior, approx_post, z_sample, rec_params)

        return vlb.mean().item(), rec_params

    def eval_iw(self, x, k=50):
        self.set_mode_train(train=False)
        x,  = variable_to_tensor_list(variables=(x, ), cuda=self.cuda)
        z_sample = self.base_VAEAC.recognition_encode(x).sample()
        approx_post = self.model.recognition_encode(z_sample)

        iw_lb = self.model.iwlb(self.prior, approx_post, z_sample, k)
        return iw_lb.mean().item()

    def recongnition(self, x, grad=False):
        self.set_mode_train(train=False)
        if grad:
            if not x.requires_grad:
                x.requires_grad = True
        else:
            x, = variable_to_tensor_list(variables=(x,), volatile=True, cuda=self.cuda)
        approx_post = self.model.encode(x)
        return approx_post

    def regenerate(self, z, grad=False):
        self.set_mode_train(train=False)
        if grad:
            if not z.requires_grad:
                z.requires_grad = True
        else:
            z, = variable_to_tensor_list(variables=(z,), volatile=True, cuda=self.cuda)
        out = self.model.decode(z)
        return out.data

    def u_recongnition(self, x, grad=False):
        self.set_mode_train(train=False)
        if grad:
            if not x.requires_grad:
                x.requires_grad = True
        else:
            x, = variable_to_tensor_list(variables=(x,), volatile=True, cuda=self.cuda)

        z = self.base_VAEAC.recognition_encode(x).loc
        approx_post = self.model.encode(z)
        return approx_post

    def u_mask_recongnition(self, x, mask, grad=False):
        self.set_mode_train(train=False)
        if grad:
            if not x.requires_grad:
                x.requires_grad = True
            mask, = variable_to_tensor_list(variables=(mask, ), cuda=self.cuda)
        else:
            x, mask = variable_to_tensor_list(variables=(x, mask), cuda=self.cuda)

        z = self.base_VAEAC.prior_encode(x, mask).loc
        approx_post = self.model.encode(z)
        return approx_post

    def u_regenerate(self, u, grad=False):
        self.set_mode_train(train=False)
        if grad:
            if not u.requires_grad:
                u.requires_grad = True
        else:
            u, = variable_to_tensor_list(variables=(u,), volatile=True, cuda=self.cuda)

        z = self.model.decode(u)
        out = self.base_VAEAC.decode(z)
        if self.base_VAEAC.pred_sig:
            return normal_parse_params(out, 1e-2)
        else:
            return out.data


class VAE_gauss_net(BaseNet):
    def __init__(self, input_dim, width, depth, latent_dim, pred_sig=True, lr=1e-3, cuda=True):
        super(VAE_gauss_net, self).__init__()
        print('y', 'VAE_gauss_net')

        self.cuda = cuda

        self.input_dim = input_dim
        self.width = width
        self.depth = depth
        self.latent_dim = latent_dim
        self.lr = lr
        self.pred_sig = pred_sig

        self.create_net()
        self.create_opt()
        self.epoch = 0
        self.schedule = None

        if self.cuda:
            self.prior = self.prior = Normal(loc=torch.zeros(latent_dim).cuda(), scale=torch.ones(latent_dim).cuda())
        else:
            self.prior = Normal(loc=torch.zeros(latent_dim), scale=torch.ones(latent_dim))
        self.vlb_scale = 1 / input_dim  # scale for dimensions of input so we can use same LR always

    def create_net(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        self.model = VAE_gauss(self.input_dim, self.width, self.depth, self.latent_dim, self.pred_sig)
        if self.cuda:
            self.model = self.model.cuda()
            cudnn.benchmark = True
        print('    Total params: %.2fM' % (self.get_nb_parameters() / 1000000.0))

    def create_opt(self):
        self.optimizer = RAdam(self.model.parameters(), lr=self.lr)

    def fit(self, x):
        self.set_mode_train(train=True)

        x, = variable_to_tensor_list(variables=(x,), cuda=self.cuda)
        self.optimizer.zero_grad()

        approx_post = self.model.encode(x)
        z_sample = approx_post.rsample()
        rec_params = self.model.decode(z_sample)

        vlb = self.model.vlb(self.prior, approx_post, x, rec_params)
        loss = (- vlb * self.vlb_scale).mean()

        loss.backward()
        self.optimizer.step()

        return vlb.mean().item(), rec_params

    def eval(self, x, sample=False):
        self.set_mode_train(train=False)

        x, = variable_to_tensor_list(variables=(x,), cuda=self.cuda)
        approx_post = self.model.encode(x)
        if sample:
            z_sample = approx_post.sample()
        else:
            z_sample = approx_post.loc
        rec_params = self.model.decode(z_sample)

        vlb = self.model.vlb(self.prior, approx_post, x, rec_params)

        return vlb.mean().item(), rec_params

        self.set_mode_train(train=False)
        x, = to_variable(var=(x,), cuda=self.cuda)

        approx_post = self.model.recognition_encode(x)

        iw_lb = self.model.iwlb(self.prior, approx_post, x, k)
        return iw_lb.mean().item()

    def recongnition(self, x, grad=False):
        self.set_mode_train(train=False)
        if grad:
            if not x.requires_grad:
                x.requires_grad = True
        else:
            x, = variable_to_tensor_list(variables=(x,), volatile=True, cuda=self.cuda)
        approx_post = self.model.encode(x)
        return approx_post

    def regenerate(self, z, grad=False):
        self.set_mode_train(train=False)
        if grad:
            if not z.requires_grad:
                z.requires_grad = True
        else:
            z, = variable_to_tensor_list(variables=(z,), volatile=True, cuda=self.cuda)
        out = self.model.decode(z)
        if self.pred_sig:
            return normal_parse_params(out, 1e-2)
        else:
            return out
