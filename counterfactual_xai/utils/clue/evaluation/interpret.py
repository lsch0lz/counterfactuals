import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from counterfactual_xai.methods.clue import decompose_std_gauss, decompose_entropy_cat
from counterfactual_xai.methods.utils import get_num_batches
from counterfactual_xai.utils.base_model import BaseNet
from counterfactual_xai.utils.clue.bnn.utils import variable_to_tensor_list
from counterfactual_xai.utils.clue.evaluation.utils import sample_artificial_targets_cat, sample_artificial_targets_gauss
from counterfactual_xai.utils.clue.vae.models import RMSCatLoglike
from counterfactual_xai.utils.clue.vae.utils import flat_to_gauss_cat


def MNIST_mean_std_norm(x):
    mean = 0.1307
    std = 0.3081
    x = x - mean
    x = x / std
    return x


def generate_ind_batch(nb_samples, batch_size, random=True, roundup=True):
    if random:
        ind = np.random.permutation(nb_samples)
    else:
        ind = range(int(nb_samples))
    for i in range(int(get_num_batches(nb_samples, batch_size, roundup))):
        yield ind[i * batch_size: (i + 1) * batch_size]


def get_BNN_uncertainties(BNN, explanations, regression, batch_size=1024, norm_MNIST=False, flatten=False, return_probs=False, prob_BNN=True):
    total_stack = []
    aleatoric_stack = []
    epistemic_stack = []
    probs_stack = []
    aux_loader = generate_ind_batch(explanations.shape[0], batch_size=batch_size, random=False, roundup=True)
    for idxs in aux_loader:
        if regression:

            if prob_BNN:
                mu_vec, std_vec = BNN.sample_predict(explanations[idxs], Nsamples=0, grad=False)
                total_uncertainty, aleatoric_uncertainty, epistemic_uncertainty = decompose_std_gauss(mu_vec, std_vec)
                probs_stack.append(mu_vec)
            else:
                mu, std = BNN.predict(explanations[idxs], grad=False)
                probs_stack.append(mu)
                total_uncertainty = std
                aleatoric_uncertainty = std
                epistemic_uncertainty = std * 0
        else:

            if norm_MNIST:
                to_BNN = MNIST_mean_std_norm(explanations[idxs])
            else:
                to_BNN = explanations[idxs]

            if flatten:
                to_BNN = to_BNN.view(to_BNN.shape[0], -1)

            if prob_BNN:
                probs = BNN.sample_predict(to_BNN, Nsamples=0, grad=False)
                total_uncertainty, aleatoric_uncertainty, epistemic_uncertainty = decompose_entropy_cat(probs)
                probs_stack.append(probs)
            else:
                probs = BNN.predict(to_BNN, grad=False)
                total_uncertainty = -(probs * torch.log(probs + 1e-10)).sum(dim=1, keepdim=False)
                aleatoric_uncertainty = total_uncertainty
                epistemic_uncertainty = total_uncertainty * 0
                probs_stack.append(probs)

        total_stack.append(total_uncertainty)
        aleatoric_stack.append(aleatoric_uncertainty)
        epistemic_stack.append(epistemic_uncertainty)

    total_stack = torch.cat(total_stack, dim=0)
    aleatoric_stack = torch.cat(aleatoric_stack, dim=0)
    epistemic_stack = torch.cat(epistemic_stack, dim=0)
    probs_stack = torch.cat(probs_stack, dim=1)
    if return_probs:
        return total_stack, aleatoric_stack, epistemic_stack, probs_stack
    else:
        return total_stack, aleatoric_stack, epistemic_stack


def evaluate_aleatoric_explanation_cat(VAEAC, explanations, test_dims, N_target_samples=500, batch_size=1024):
    """This assumes that test dims are placed at end of input vector"""
    explanations = explanations.view(explanations.shape[0], -1)
    explanations_expand = torch.cat([explanations, \
                                     explanations.new_zeros((explanations.shape[0], len(test_dims)))], dim=1)
    assert explanations_expand.shape[1] == (test_dims[-1] + 1)

    output_stack = []
    aux_loader = generate_ind_batch(explanations_expand.shape[0], batch_size=batch_size, random=False, roundup=True)
    for idxs in aux_loader:
        probs = sample_artificial_targets_cat(VAEAC, explanations_expand[idxs], test_dims, N_target_samples).data
        output_stack.append(probs.data.cpu())

    y_cond_explan_aleatoric_probs = torch.cat(output_stack, dim=1)
    y_cond_explan_aleatoric_entropy, _, _ = decompose_entropy_cat(y_cond_explan_aleatoric_probs)
    return y_cond_explan_aleatoric_entropy


def evaluate_aleatoric_explanation_gauss(VAEAC, explanations, test_dims, pred_sig, N_target_samples=500, batch_size=1024):
    """This assumes that test dims are placed at end of input vector"""
    explanations = explanations.view(explanations.shape[0], -1)
    explanations_expand = torch.cat([explanations, \
                                     explanations.new_zeros((explanations.shape[0], len(test_dims)))], dim=1)
    assert explanations_expand.shape[1] == (test_dims[-1] + 1)

    output_means_stack = []
    output_stds_stack = []
    aux_loader = generate_ind_batch(explanations_expand.shape[0], batch_size=batch_size, random=False, roundup=True)
    for idxs in aux_loader:
        means, stds = sample_artificial_targets_gauss(VAEAC, explanations_expand[idxs], test_dims, N_target_samples,
                                                      pred_sig, z_mean=False)
        output_means_stack.append(means.data.cpu())
        output_stds_stack.append(stds.data.cpu())

    y_cond_explan_aleatoric_means = torch.cat(output_means_stack, dim=1)
    y_cond_explan_aleatoric_stds = torch.cat(output_stds_stack, dim=1)
    y_cond_explan_aleatoric_entropy, _, _ = decompose_std_gauss(y_cond_explan_aleatoric_means, y_cond_explan_aleatoric_stds)
    return y_cond_explan_aleatoric_entropy


def evaluate_BNN_epistemic_class_error_loglike_cat(BNN, epistemic_explanations,
                                                   explanation_targets, batch_size=1024, flatten=False):
    if flatten:
        epistemic_explanations = epistemic_explanations.view(epistemic_explanations.shape[0], -1)

    aux_loader = generate_ind_batch(epistemic_explanations.shape[0], batch_size, random=False, roundup=True)
    loglike_vec = []
    test_err = 0
    nb_samples = 0
    for idxs in aux_loader:
        probs_samples = BNN.sample_predict(x=epistemic_explanations[idxs], Nsamples=0, grad=False).data
        probs = probs_samples.mean(dim=0)

        log_probs = torch.log(probs)
        loss = F.nll_loss(log_probs, explanation_targets[idxs], reduction='none').data

        pred = probs.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        err = pred.ne(explanation_targets[idxs].data).sum()

        loglike_vec.append(-loss)
        test_err += err.cpu().numpy()
        nb_samples += len(idxs)

    test_err /= nb_samples
    loglike_vec = torch.cat(loglike_vec, dim=0)

    return test_err, loglike_vec


def evaluate_epistemic_explanation_cat(BNN, VAEAC, epistemic_explanation, test_dims, outer_batch_size=2000,
                                       inner_batch_size=1024, VAEAC_samples=500):
    epistemic_explanation = epistemic_explanation.view(epistemic_explanation.shape[0], -1)

    epistemic_explanation_expand = torch.cat(
        [epistemic_explanation, epistemic_explanation.new_zeros(epistemic_explanation.shape[0], len(test_dims))], dim=1)

    output_stack = []
    aux_loader = generate_ind_batch(epistemic_explanation_expand.shape[0], outer_batch_size, random=False, roundup=True)
    for idxs in aux_loader:
        probs = sample_artificial_targets_cat(VAEAC, epistemic_explanation_expand[idxs],
                                              test_dims=test_dims, N_target_samples=VAEAC_samples, z_mean=False,
                                              softmax=False).data
        output_stack.append(probs.cpu())
    # We have the VAEAC give us its expected prediction
    y_cond_explanation_epistemic_probs = torch.cat(output_stack, dim=1).mean(dim=0)
    y_cond_explanation_epistemic_preds = y_cond_explanation_epistemic_probs.max(dim=1)[1]

    test_err, loglike_vec = evaluate_BNN_epistemic_class_error_loglike_cat(BNN, epistemic_explanation,
                                                                           y_cond_explanation_epistemic_preds.cuda(),
                                                                           batch_size=inner_batch_size, flatten=False)
    loglike_vec = loglike_vec.cpu().numpy()
    return test_err, loglike_vec


def evaluate_BNN_epistemic_err_gauss(BNN, epistemic_explanations, y_cond_explanations_mu_epistemic,
                                     batch_size=1000, flatten=False):
    # Nte that this function does not unnormalise
    if flatten:
        epistemic_explanations = epistemic_explanations.view(epistemic_explanations.shape[0], -1)

    aux_loader = generate_ind_batch(epistemic_explanations.shape[0], batch_size, random=False, roundup=True)
    diffs = []
    for idxs in aux_loader:
        mu, std = BNN.sample_predict(x=epistemic_explanations[idxs], Nsamples=0, grad=False)
        pred = mu.mean(dim=0).cpu()
        diffs.append((y_cond_explanations_mu_epistemic[idxs] - pred).abs().data)
    diffs = torch.cat(diffs, dim=0)

    rms = torch.sqrt((diffs ** 2).sum() / diffs.shape[0])

    return rms, diffs


def evaluate_epistemic_explanation_gauss(BNN, VAEAC, epistemic_explanation, test_dims, pred_sig, outer_batch_size=2000,
                                         inner_batch_size=1024, VAEAC_samples=500):
    epistemic_explanation = epistemic_explanation.view(epistemic_explanation.shape[0], -1)

    epistemic_explanation_expand = torch.cat(
        [epistemic_explanation, epistemic_explanation.new_zeros(epistemic_explanation.shape[0], len(test_dims))], dim=1)

    output_means_stack = []
    aux_loader = generate_ind_batch(epistemic_explanation_expand.shape[0], outer_batch_size, random=False, roundup=True)
    for idxs in aux_loader:
        y_cond_x_mu, y_cond_x_std = sample_artificial_targets_gauss(VAEAC, epistemic_explanation_expand[idxs], pred_sig=pred_sig,
                                                                    test_dims=test_dims, N_target_samples=VAEAC_samples, z_mean=False)
        output_means_stack.append(y_cond_x_mu.data.cpu())
    # Get the expected prediction from the BNN
    y_cond_explanation_epistemic_mu = torch.cat(output_means_stack, dim=1).mean(dim=0)

    rms, diffs = evaluate_BNN_epistemic_err_gauss(BNN, epistemic_explanation, y_cond_explanation_epistemic_mu,
                                                  batch_size=inner_batch_size, flatten=False)
    return rms.cpu().numpy(), diffs.cpu().numpy()

def get_VAEAC_px(under_VAEAC_net, x_art_test, y_dims, Nsamples=5000, bern=False, batch_size=None):
    """Note that this function automatically masks y and only takes x as input
        Works for factorised Bernouilli inputs and Gaussian inputs"""
    max_dims = x_art_test.shape[1] + len(y_dims)
    x_dims = range(max_dims)
    for e in y_dims:
        x_dims.remove(e)

    iw_xy = torch.zeros((x_art_test.shape[0], max_dims))
    iw_xy[:, x_dims] = torch.Tensor(x_art_test)
    iw_xy[:, y_dims] = iw_xy.new_zeros((iw_xy.shape[0], len(y_dims)))

    iw_mask = torch.zeros_like(iw_xy)
    iw_mask[:, y_dims] = 1

    prior = under_VAEAC_net.prior
    # batching from here
    log_px_vec = []

    if batch_size is None:
        batch_size = iw_xy.shape[0]
    aux_loader = generate_ind_batch(iw_xy.shape[0], batch_size, random=False, roundup=True)
    for idxs in aux_loader:

        u_approx_dist = under_VAEAC_net.u_mask_recongnition(iw_xy[idxs], iw_mask[idxs], grad=False)

        p_x_estimates = []
        for i in range(Nsamples):

            u_sample = u_approx_dist.sample().data
            rec_distrib = under_VAEAC_net.u_regenerate(u_sample, grad=False)

            log_p = prior.log_prob(u_sample).sum(dim=1).data
            log_q = u_approx_dist.log_prob(u_sample).sum(dim=1).data
            if bern:
                rec_loglike = -F.binary_cross_entropy_with_logits(rec_distrib, iw_xy[idxs].type(rec_distrib.type()), reduction='none')
            else:
                rec_loglike = rec_distrib.log_prob(iw_xy[idxs].type(u_sample.type())).data
            x_loglike = rec_loglike[:, x_dims].sum(dim=1)

            p_x_estimates.append(x_loglike + log_p - log_q)

        log_px = torch.logsumexp(torch.stack(p_x_estimates), dim=0, keepdim=False) - np.log(Nsamples)

        log_px_vec.append(log_px)

    log_px_vec = torch.cat(log_px_vec, dim=0)
    return log_px_vec

def get_VAEAC_px_gauss_cat(under_VAEAC_net, x_art_test, input_dim_vec, y_dims, override_y_dims=None, Nsamples=5000):
    """Note that this function automatically masks y in generations and only takes x as input"""
    rec_loglike_func = RMSCatLoglike(input_dim_vec, reduction='none')
    max_dims = x_art_test.shape[1] + len(y_dims)
    x_dims = range(max_dims)
    for e in y_dims:
        x_dims.remove(e)

    iw_xy = torch.zeros((x_art_test.shape[0], max_dims))
    iw_xy[:, x_dims] = torch.Tensor(x_art_test)
    iw_xy[:, y_dims] = iw_xy.new_zeros((iw_xy.shape[0], len(y_dims))).normal_()  # this offsets for the max operation
    if under_VAEAC_net.cuda:
        iw_xy = iw_xy.cuda()

    iw_xy_target = flat_to_gauss_cat(iw_xy, input_dim_vec)

    iw_mask = torch.zeros_like(iw_xy)
    iw_mask[:, y_dims] = 1

    prior = under_VAEAC_net.prior
    u_approx_dist = under_VAEAC_net.u_mask_recongnition(iw_xy, iw_mask, grad=False)

    p_x_estimates = []
    for i in range(Nsamples):

        u_sample = u_approx_dist.sample()
        rec_distrib = under_VAEAC_net.u_regenerate(u_sample, grad=False)

        log_p = prior.log_prob(u_sample).sum(dim=1).data
        log_q = u_approx_dist.log_prob(u_sample).sum(dim=1).data
        rec_loglike = rec_loglike_func(rec_distrib, iw_xy_target).view(iw_xy.shape[0], -1)

        if override_y_dims is not None:
            x_loglike = rec_loglike[:, :-override_y_dims].sum(dim=1)
        else:
            x_loglike = rec_loglike[:, x_dims].sum(dim=1)
        p_x_estimates.append(x_loglike + log_p - log_q)

    log_px = torch.logsumexp(torch.stack(p_x_estimates), dim=0, keepdim=False) - np.log(Nsamples)
    return log_px

def input_uncertainty_step_gauss(BNN, dset, aleatoric_coeff, epistemic_coeff, stepsize_perdim=-1,
                                 batch_size=1024, cuda=True, entropy=False, norm_grad=False):
    """Takes a single step in the direction of uncertainty gradient wrt input"""
    if cuda:
        trainloader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                                  num_workers=3)
    else:
        trainloader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False, pin_memory=False,
                                                  num_workers=3)
    output_x = []
    for x, y in trainloader:
        x, = variable_to_tensor_list(variables=(x,), cuda=cuda)

        mu_vec, std_vec = BNN.sample_predict(x, Nsamples=0, grad=True)

        if entropy:
            raise Exception('Deprecated option, will remove soon')
            # total_uncert, aleatoric_uncert, epistemic_uncert = decompose_entropy_N_gauss(mu_vec, std_vec)
        else:
            total_uncert, aleatoric_uncert, epistemic_uncert = decompose_std_gauss(mu_vec, std_vec)

        objective = aleatoric_coeff * aleatoric_uncert.sum() + epistemic_coeff * epistemic_uncert.sum()
        objective.backward()

        if norm_grad:
            l1_norm_step_dir = x.grad / (torch.abs(x.grad).sum(dim=1, keepdim=True) / x.shape[1] + 1e-12)
        else:
            l1_norm_step_dir = x.grad

        new_x = x + stepsize_perdim * l1_norm_step_dir  # gradient descent is induced by negative coeff in function input
        output_x.append(new_x)

    output_x = torch.cat(output_x)
    return output_x

def input_uncertainty_step_cat(BNN, dset, aleatoric_coeff, epistemic_coeff, stepsize_perdim=-1,
                                 batch_size=1024, cuda=True, norm_MNIST=False, flatten=False, norm_grad=False):
    """Takes a single step in the direction of uncertainty gradient wrt input"""
    if cuda:
        trainloader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                                  num_workers=3)
    else:
        trainloader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False, pin_memory=False,
                                                  num_workers=3)
    output_x = []
    for x, y in trainloader:
        x, = variable_to_tensor_list(variables=(x,), cuda=cuda)
        x.requires_grad = True

        if norm_MNIST:
            to_BNN = MNIST_mean_std_norm(x)
        else:
            to_BNN = x

        if flatten:
            to_BNN = to_BNN.view(to_BNN.shape[0], -1)

        probs = BNN.sample_predict(to_BNN, Nsamples=0, grad=True)
        _, aleatoric_uncert, epistemic_uncert = decompose_entropy_cat(probs)
        objective = aleatoric_coeff * aleatoric_uncert.sum() + epistemic_coeff * epistemic_uncert.sum()
        objective.backward()

        if norm_grad:
            l1_norm_step_dir = x.grad / (torch.abs(x.grad).sum(dim=1, keepdim=True) / x.shape[1] + 1e-12)
        else:
            l1_norm_step_dir = x.grad

        new_x = x + stepsize_perdim * l1_norm_step_dir  # gradient descent is induced by negative coeffs
        output_x.append(new_x)

    output_x = torch.cat(output_x)
    return output_x

def gumbel_sigmoid(prob_map, temperature, eps=1e-20):
    U = prob_map.new(prob_map.shape).uniform_(0, 1)
    sigmoid_in = torch.log(prob_map + eps) - torch.log(1 - prob_map + eps) + torch.log(U + eps) - torch.log(1 - U + eps)
    y = torch.sigmoid(sigmoid_in / temperature)
    y_hard = torch.round(y)
    return (y_hard - y).detach() + y

class bern_mask(nn.Module):
    def __init__(self, shape, init_p=0.5, temp=0.1):
        super(bern_mask, self).__init__()

        self.mask_probs = nn.Parameter(torch.ones(shape) * init_p)
        self.temp = temp

    def forward(self, x):
        hard_mask = gumbel_sigmoid(self.mask_probs, self.temp)
        return x * hard_mask, (1 - hard_mask)


class mask_explainer(BaseNet):
    def __init__(self, shape, mask_L1_weight, aleatoric_coeff, epistemic_coeff,
                 mask_samples=1, lr=0.05, decay_period=5, gamma=0.8, cuda=True):
        super(mask_explainer, self).__init__()

        self.mask_L1_weight = mask_L1_weight
        self.aleatoric_coeff = aleatoric_coeff
        self.epistemic_coeff = epistemic_coeff
        self.mask_samples = mask_samples

        self.model = bern_mask(shape, init_p=0.5, temp=0.1)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=decay_period, gamma=gamma)

        self.cuda = cuda
        if self.cuda:
            self.model = self.model.cuda()

    def fit_cat(self, x, BNN, VAEAC, flatten_ims=True, test_dims=None, plot=False):
        # note that x will need to be the same shape as specified at class initialisation
        x, = variable_to_tensor_list(variables=(x,), cuda=self.cuda)
        self.set_mode_train(train=True)
        BNN.set_mode_train(train=False)
        VAEAC.set_mode_train(train=False)

        self.optimizer.zero_grad()
        loss_cum = 0
        aleatoric_cum = 0
        epistemic_cum = 0

        for it in range(self.mask_samples):

            masked_x, mask = self.model(x)

            if flatten_ims:
                flat_x = x.view(masked_x.shape[0], -1)
                masked_x = masked_x.view(masked_x.shape[0], -1)
                mask = mask.view(mask.shape[0], -1)
            else:
                flat_x = x
            if test_dims is not None:
                #                 x = torch.cat([x, x.new_zeros(x.shape[0], test_dims)], dim=1)
                masked_x = torch.cat([masked_x, masked_x.new_zeros(masked_x.shape[0], test_dims)], dim=1)
                mask = torch.cat([mask, mask.new_ones(mask.shape[0], test_dims)], dim=1)

            # We dont want gradients from this
            inpainted = VAEAC.inpaint(masked_x.data, mask.data, Nsample=1, z_mean=True).data.squeeze(0)
            if test_dims is not None:
                inpainted = inpainted[:, :-test_dims]
                mask = mask[:, :-test_dims]

            to_BNN = inpainted * mask + flat_x * (1 - mask)
            to_BNN = MNIST_mean_std_norm(to_BNN)

            probs = BNN.sample_predict(to_BNN, Nsamples=0, grad=True)
            total_entropy, aleatoric_entropy, epistemic_entropy = decompose_entropy_cat(probs)

            # We mean across batch
            aleatoric_cum += aleatoric_entropy.mean().item()
            epistemic_cum += epistemic_entropy.mean().item()
            # we should average over MC samples but sum over batch and features
            loss = (self.aleatoric_coeff * aleatoric_entropy +
                    self.epistemic_coeff * epistemic_entropy + self.mask_L1_weight * mask.sum(dim=1)).sum(dim=0) / self.mask_samples
            loss.backward()  # Gradient accumulation
            loss_cum += loss.item() / aleatoric_entropy.shape[0]

        # we average here to be invariant to number of samples
        aleatoric_cum = aleatoric_cum / self.mask_samples
        epistemic_cum = epistemic_cum / self.mask_samples

        self.optimizer.step()
        self.model.mask_probs.data = torch.clamp(self.model.mask_probs, min=0, max=1)
        self.scheduler.step()

        return loss_cum, aleatoric_cum, epistemic_cum

    def fit_gauss(self, x, BNN, VAEAC, flatten_ims=True, test_dims=None, plot=False):
        x, = variable_to_tensor_list(variables=(x,), cuda=self.cuda)
        self.set_mode_train(train=True)
        BNN.set_mode_train(train=False)
        VAEAC.set_mode_train(train=False)

        self.optimizer.zero_grad()
        loss_cum = 0
        aleatoric_cum = 0
        epistemic_cum = 0

        for it in range(self.mask_samples):

            masked_x, mask = self.model(x)

            if flatten_ims:
                flat_x = x.view(masked_x.shape[0], -1)
                masked_x = masked_x.view(masked_x.shape[0], -1)
                mask = mask.view(mask.shape[0], -1)
            else:
                flat_x = x
            if test_dims is not None:
                #                 x = torch.cat([x, x.new_zeros(x.shape[0], test_dims)], dim=1)
                masked_x = torch.cat([masked_x, masked_x.new_zeros(masked_x.shape[0], test_dims)], dim=1)
                mask = torch.cat([mask, mask.new_ones(mask.shape[0], test_dims)], dim=1)

            # We dont want gradients from this
            # Switched to non gauss output
            inpainted = VAEAC.inpaint(masked_x.data, mask.data, Nsample=1, z_mean=True).data.squeeze(0)
            if test_dims is not None:
                inpainted = inpainted[:, :-test_dims]
                mask = mask[:, :-test_dims]

            to_BNN = inpainted * mask + flat_x * (1 - mask)

            mu, std = BNN.sample_predict(to_BNN, Nsamples=0, grad=True)
            total_std, aleatoric_std, epistemic_std = decompose_std_gauss(mu, std)

            # we average here to be invariant to batch size
            aleatoric_cum += aleatoric_std.mean().item()
            epistemic_cum += epistemic_std.mean().item()
            loss = (self.aleatoric_coeff * aleatoric_std +
                    self.epistemic_coeff * epistemic_std + self.mask_L1_weight * mask.sum(dim=1)).sum(dim=0) / self.mask_samples
            loss.backward()  # Gradient accumulation
            loss_cum += loss.item() / aleatoric_std.shape[0]

        # we average here to be invariant to number of samples
        aleatoric_cum = aleatoric_cum / self.mask_samples
        epistemic_cum = epistemic_cum / self.mask_samples

        #         loss_cum.backward()
        self.optimizer.step()
        self.model.mask_probs.data = torch.clamp(self.model.mask_probs, min=0, max=1)
        self.scheduler.step()

        return loss_cum, aleatoric_cum, epistemic_cum

    def get_mask(self):
        self.set_mode_train(train=False)
        """Note that this returns 1s for input features which are masked"""
        return 1 - self.model.mask_probs.data.round()

    def get_mask_probs(self):
        self.set_mode_train(train=False)
        """Note that this returns 1s for input features which are masked"""
        return 1 - self.model.mask_probs.data

    def mask_input(self, x):
        self.set_mode_train(train=False)
        x, = variable_to_tensor_list(variables=(x,), cuda=self.cuda)
        self.set_mode_train(train=False)
        return x * self.model.mask_probs.data.round()

    def mask_inpaint(self, x, VAEAC, flatten_ims=True, test_dims=None, cat=False):
        x, = variable_to_tensor_list(variables=(x,), cuda=self.cuda)
        self.set_mode_train(train=False)
        VAEAC.set_mode_train(train=False)

        masked_x = x * self.model.mask_probs.data.round().data
        mask = 1 - self.model.mask_probs.data.round().data

        if flatten_ims:
            flat_x = x.view(masked_x.shape[0], -1)
            masked_x = masked_x.view(masked_x.shape[0], -1)
            mask = mask.view(mask.shape[0], -1)
        else:
            flat_x = x

        if test_dims is not None:
            masked_x = torch.cat([masked_x, masked_x.new_zeros(masked_x.shape[0], test_dims)], dim=1)
            mask = torch.cat([mask, mask.new_ones(mask.shape[0], test_dims)], dim=1)

        # We dont want gradients from this
        if cat:
            inpainted = VAEAC.inpaint(masked_x.data, mask.data, Nsample=1, z_mean=True).data.squeeze(0)
        else:
            inpainted = VAEAC.inpaint(masked_x.data, mask.data, Nsample=1, z_mean=True).data.squeeze(0)
        if test_dims is not None:
            inpainted = inpainted[:, :-test_dims]
            mask = mask[:, :-test_dims]

        out = inpainted * mask + flat_x * (1 - mask)

        return out.data, mask.data

    @staticmethod
    def train_mask(x, BNN, VAEAC, aleatoric_coeff, epistemic_coeff, L1w=1, N_epochs=30,
                   mask_samples=20, mask_samples2=10, cat=True, flatten_ims=True, test_dims=None):

        torch.cuda.empty_cache()
        x_pixels = x.view(x.shape[0], -1).shape[1]

        explainer = mask_explainer(shape=x.shape, mask_L1_weight=L1w / x_pixels, aleatoric_coeff=aleatoric_coeff,
                                   epistemic_coeff=epistemic_coeff, mask_samples=mask_samples, lr=0.05, decay_period=5,
                                   gamma=0.8, cuda=True)

        loss_vec = []
        aleatoric_vec = []
        epistemic_vec = []
        for i in range(N_epochs):
            if i > 10:  # We train with more samples at the beginning as there is a lot more variability
                explainer.mask_samples = mask_samples2
            if cat:
                loss, aleatoric_ent, epistemic_ent = explainer.fit_cat(x, BNN, VAEAC, flatten_ims=flatten_ims,
                                                                       test_dims=test_dims, plot=False)
            else:
                loss, aleatoric_ent, epistemic_ent = explainer.fit_gauss(x, BNN, VAEAC, flatten_ims=flatten_ims,
                                                                         test_dims=test_dims, plot=False)

            loss_vec.append(loss)
            aleatoric_vec.append(aleatoric_ent)
            epistemic_vec.append(epistemic_ent)
            print('it: %d, loss: %3.3f, aleatoric: %3.3f, epistemic: %3.3f' % (i, loss, aleatoric_ent, epistemic_ent))

        loss_vec = np.array(loss_vec)
        aleatoric_vec = np.array(aleatoric_vec)
        epistemic_vec = np.array(epistemic_vec)

        return explainer, loss_vec, aleatoric_vec, epistemic_vec