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
            cum_dims += 1
        elif dim > 1:
            if not cat_probs:
                if not grad:
                    y = x[:, cum_dims:cum_dims + dim].max(dim=1)[1]
                    y_vec = torch_onehot(y, dim).type(x.type())
                    output[:, cum_dims:cum_dims + dim] = y_vec
                else:
                    x_cat = x[:, cum_dims:cum_dims + dim]
                    probs = F.softmax(x_cat, dim=1)
                    y_hard = x[:, cum_dims:cum_dims + dim].max(dim=1)[1]
                    y_oh = torch_onehot(y_hard, dim).type(x.type())
                    output[:, cum_dims:cum_dims + dim] = (y_oh - probs).detach() + probs
            else:
                x_cat = x[:, cum_dims:cum_dims + dim]
                probs = F.softmax(x_cat, dim=1)

                if prob_sample:  # we are going to use gumbel trick here
                    log_probs = torch.log(probs)
                    u = log_probs.new(log_probs.shape).uniform_(0, 1)
                    g = -torch.log(-torch.log(u + eps) + eps)
                    cat_samples = (log_probs + g).max(dim=1)[1]
                    hard_samples = torch_onehot(cat_samples, dim).type(x.type())
                    output[:, cum_dims:cum_dims + dim] = hard_samples
                else:
                    output[:, cum_dims:cum_dims + dim] = probs


            cum_dims += dim
        else:
            raise ValueError('Error, invalid dimension value')
    return output
