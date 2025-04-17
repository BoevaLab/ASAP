from functools import partial

import numpy as np
from matplotlib import pyplot as plt, style
from scipy.stats import pearsonr, spearmanr

from src.snv.plot_PCC_vs_reads import calculate_score

style.use('seaborn-v0_8-colorblind')


def plot_fc(df, wgs_vaf_lims: tuple = (0, 1), read_limit: int = 5, approx_true: bool = True, s=8,
            logscale: bool = True, ref_over_var: bool = True, color_col: str = 'wgs_vaf', save: str = None, window=10):
    tmp_df = df.copy()
    tmp_df = tmp_df[(wgs_vaf_lims[0] < tmp_df.wgs_vaf) & (tmp_df.wgs_vaf < wgs_vaf_lims[1])
                    & (tmp_df.atac_ref_reads + tmp_df.atac_alt_reads >= read_limit)
                    & (tmp_df.atac_alt_reads != 0) & (tmp_df.atac_ref_reads != 0)
                    ]
    tmp_df['score'] = tmp_df.apply(partial(calculate_score, ref_over_var=ref_over_var, window=window), axis=1)

    true_ref, true_var = tmp_df.atac_ref_reads.to_numpy(), tmp_df.atac_alt_reads.to_numpy()

    if approx_true:
        true_ref /= 1 - tmp_df.wgs_vaf
        true_var /= tmp_df.wgs_vaf

    y1 = true_ref / true_var if ref_over_var else true_var / true_ref
    y2 = tmp_df.score.to_numpy()

    # r, p = spearmanr(y1, y2)
    r, p = pearsonr(y1, y2)
    print('Pearson', pearsonr(y1, y2), '\nSpearman', spearmanr(y1, y2))
    if logscale:
        y1 = np.log2(y1)
        y2 = np.log2(y2)
    fig = plt.figure(figsize=(4, 3), dpi=200)

    color_arr = tmp_df[color_col].to_numpy()
    corrs = []
    if not np.issubdtype(color_arr.dtype, np.number):
        all_xs = []
        all_ys = []
        for unique_el in np.unique(color_arr):
            idx = (tmp_df[color_col] == unique_el).to_numpy()
            corr = pearsonr(y1[idx], y2[idx])
            print(f'Corr for {unique_el}: {corr}')
            corrs.append(corr[0])
            plt.scatter(y1[idx], y2[idx], label=unique_el, s=s)
            all_xs.append(y1[idx])
            all_ys.append(y2[idx])

        for i in range(len(all_xs[0])):
            xis = [x[i] for x in all_xs]
            yis = [y[i] for y in all_ys]
            plt.plot(xis, yis, 'k-', linewidth=0.5)
    else:
        plt.scatter(y1, y2, label=f'PCC={r:.2f}, p={p:.3f}', c=color_arr, s=s)
        plt.colorbar(label=color_col)
    print('Mean corr', np.array(corrs).mean())

    plt.ylabel(f'Predicted {"log " if logscale else ""}FC {"(ref/var)" if ref_over_var else "(var/ref)"}')
    plt.xlabel(f'Tumor ATAC reads {"log " if logscale else ""}FC {"(ref/var)" if ref_over_var else "(var/ref)"}')

    plt.axhline(0, linewidth=1, color='black', linestyle='--')
    plt.axvline(0, linewidth=1, color='black', linestyle='--')
    # plt.legend(loc='upper left', bbox_to_anchor=(0, 1.15), borderaxespad=0.)
    plt.legend(fontsize=5)

    fig.tight_layout()
    if save:
        fig.savefig(
            f'../imgs/svg/{save}_FC_{"ref:var" if ref_over_var else "var:ref"}_log-log.svg')
    plt.show()
