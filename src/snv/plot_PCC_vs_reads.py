import json
from functools import partial

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr, spearmanr

style.use('seaborn-v0_8-colorblind')


def score_vs_reads(df, wgs_vaf_lims: tuple = (0, 1), approx_true: bool = True,
                   ref_over_var: bool = True, fn: str = 'pearson', x_max=None, save: str = None, window=100):
    assert fn in ['pearson', 'spearman']
    fn = spearmanr if fn == 'spearman' else pearsonr

    tmp_df = df.copy().dropna()
    tmp_df = tmp_df[(wgs_vaf_lims[0] < tmp_df.wgs_vaf) & (tmp_df.wgs_vaf < wgs_vaf_lims[1])
                    & (tmp_df.atac_alt_reads != 0) & (tmp_df.atac_ref_reads != 0)
                    ]
    tmp_df['score'] = tmp_df.apply(partial(calculate_score, ref_over_var=ref_over_var, window=window),
                                   axis=1)
    x_max = tmp_df.atac_reads.max() if x_max is None else x_max

    n_values = np.arange(tmp_df.atac_reads.min(), x_max, 1)  # Define the step size

    scores = []
    pvals = []
    samples = []
    for n in n_values:
        tmp_df = tmp_df[tmp_df.atac_reads >= n]

        if len(tmp_df) > 1:
            true_ref, true_var = tmp_df.atac_ref_reads, tmp_df.atac_alt_reads
            # true_ref, true_var = tmp_df[experiment + '_true'] * (1 - tmp_df.ATAC_VAF), tmp_df[experiment + '_true'] * tmp_df.ATAC_VAF
            if approx_true:
                true_ref /= (1 - tmp_df.wgs_vaf)
                true_var /= tmp_df.wgs_vaf

            y2 = tmp_df.score
            if ref_over_var:
                y1 = true_ref / true_var
                # y2 = tmp_df[ref_col] / tmp_df[alt_col]
            else:
                y1 = true_var / true_ref
                # y2 = tmp_df[alt_col] / tmp_df[ref_col]

            score, pval = fn(y1, y2)
            scores.append(score)
            pvals.append(pval)
        else:
            scores.append(np.nan)
            pvals.append(np.nan)
        samples.append(len(tmp_df))

    fig, ax1 = plt.subplots(dpi=200, figsize=(6, 4))
    ax1.bar(n_values, samples, alpha=0.3, color='C0')
    ax1.set_ylabel('Nr samples', color='C0')

    ax2 = ax1.twinx()
    ax2.step(n_values, pvals, color='C2', where='mid', linestyle='dashed')
    ax2.set_ylabel('p-val', color='C2')
    ax2.spines['right'].set_position(('axes', 1.15))

    ax3 = ax1.twinx()
    ax3.step(n_values, scores, color='C1', where='mid')
    ax3.set_ylabel('PCC', color='C1')
    # plt.title(f'FC {"(ref/var)" if ref_over_var else "(var/ref)"} accuracy dependence on min ATAC reads')
    ax1.set_xlabel('min total ATAC reads per sample')
    plt.grid()

    fig.tight_layout()
    if save:
        fig.savefig(
            f'../imgs/svg/{save}_FC_{"ref:var" if ref_over_var else "var:ref"}_PCC vs ATAC reads.svg')

    plt.show()


def smooth(y, sigma=8):
    return gaussian_filter1d(np.array(y), sigma)


def calculate_score(row, ref_over_var: bool, window: int = 1000):
    window = window // 5  # 5bp

    ref_list = json.loads(row['signal_pred_ref'])
    alt_list = json.loads(row['signal_pred_alt'])

    # smooth
    ref_list = smooth(ref_list)
    alt_list = smooth(alt_list)

    # take diff over window
    n = len(ref_list)
    start, end = (n // 2 - window // 2), (n // 2 + window // 2)
    ref_list = ref_list[start:end]
    alt_list = alt_list[start:end]

    if ref_over_var:
        score = sum(ref_list) / sum(alt_list)
    else:
        score = sum(alt_list) / sum(ref_list)
    return score

