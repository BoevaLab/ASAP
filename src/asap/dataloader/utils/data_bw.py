import numpy as np

from .io import get_bw_from_file

def get_coverage(bw, chrom, start, end):
    chroms_in_bw = bw.chroms()
    use_chr = any(c.startswith("chr") for c in chroms_in_bw)
    chrom_query = f'chr{chrom}' if use_chr else f'{chrom}'
    return bw.values(chrom_query, start, end, numpy=True).astype('float16')

def get_binned_signal(bw_file: str, chrom: int, start: int, end: int, bin_size: int, binning_type: str='max'):
    print(f'Binning signal for chr{chrom}:[{start}, {end}] with mode {binning_type} using {bw_file}')
    assert (end - start) % bin_size == 0, \
        f'Difference between start ({start}) and end ({end}) must be divisible by bin size {bin_size}!'
    bw = get_bw_from_file(bw_file)
    coverage = get_coverage(bw, chrom, start, end)

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
