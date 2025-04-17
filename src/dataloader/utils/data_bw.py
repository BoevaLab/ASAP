import numpy as np

from src.dataloader.utils.io import get_bw_from_file


def get_binned_signal(bw_file: str, chrom: int, start: int, end: int, bin_size: int, binning_type: str='max'):
    print(f'Binning signal for chr{chrom}:[{start}, {end}] with mode {binning_type} using {bw_file}')
    assert (end - start) % bin_size == 0, \
        f'Difference between start ({start}) and end ({end}) must be divisible by bin size {bin_size}!'
    bw = get_bw_from_file(bw_file)
    coverage = bw.values(f'chr{chrom}', start, end, numpy=True).astype('float16')

    nans = np.isnan(coverage)
    if nans.sum() > 0:
        baseline = 0
        print(f'Replacing {nans.sum()} nans in signal with length {len(coverage)} with baseline {baseline}')
        coverage[nans] = baseline

    coverage = coverage.reshape((-1, bin_size))
    if binning_type == 'mean':
        coverage = coverage.mean(axis=1)
    else:
        coverage = coverage.max(axis=1)
    return coverage
