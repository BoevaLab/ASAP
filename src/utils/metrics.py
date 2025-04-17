import numpy as np
from scipy.stats import spearmanr, kendalltau

def compute_metrics(output_list, target_list, logspace_input=False):
    if logspace_input:
        output_list_nonlog = np.exp(output_list) - 1
        target_list_nonlog = np.exp(target_list) - 1
    else:
        output_list_nonlog = output_list
        target_list_nonlog = target_list
        output_list = np.log(output_list + 1)
        target_list = np.log(target_list + 1)

    corr_spearman, _ = spearmanr(output_list, target_list)
    corr_kendall, _ = kendalltau(output_list, target_list)
    corr_pearson = np.corrcoef(output_list, target_list)[0,1]
    corr_pearson_nonlog = np.corrcoef(output_list_nonlog, target_list_nonlog)[0,1]
    mse_ = mse(output_list, target_list)
    mse_nonlog = mse(output_list_nonlog, target_list_nonlog)
    poisson_nll_ = poisson_nll(output_list, target_list)
    poisson_nll_nonlog = poisson_nll(output_list_nonlog, target_list_nonlog)

    res = {
        'spearman_r': corr_spearman,
        'kendall_tau': corr_kendall,
        'pearson_r': corr_pearson,
        'pearson_r_nonlog': corr_pearson_nonlog,
        'mse': mse_,
        'mse_nonlog': mse_nonlog,
        'poisson_nll': poisson_nll_,
        'poisson_nll_nonlog': poisson_nll_nonlog,
    }

    return res


def mse(input, target):
    return np.mean((input - target)**2)


def poisson_nll(input, target, eps=1e-8):
    loss = input - target * np.log(input + eps)
    stirling_term = (
            target * np.log(target + eps) - target + 0.5 * np.log(2 * np.pi * target)
        )
    stirling_term[target==0] = 0
    return np.mean(loss + stirling_term)
