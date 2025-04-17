import numpy as np

from src.dataloader.utils.io import get_chrom_npz


def normalize_samples(arr: np.ndarray) -> np.ndarray:
    assert len(arr.shape) == 2
    # set each peak max height to 1
    max_peaks = np.max(arr, axis=1)
    max_peaks[max_peaks == 0] = 1
    normalized_arr = arr / max_peaks[:, np.newaxis]
    return normalized_arr


def get_ys_from_npzs(npz_files: list[str], normalize: bool = False):
    # get data for both
    dicts = [get_chrom_npz(f) for f in npz_files]
    print(dict({file: chrom_dict.keys() for file, chrom_dict in zip(npz_files, dicts)}))

    # filter to keep same chroms
    common_chroms = set(list(dicts[0].keys()))
    for chrom_dict in dicts:
        common_chroms = common_chroms.intersection(set(chrom_dict.keys()))
    common_chroms = sorted(list(common_chroms))
    print(f'Keeping chroms: {common_chroms}')

    # combine predictions across chroms
    ys = [np.concatenate([chrom_dict[chrom] for chrom in common_chroms]) for chrom_dict in dicts]
    print(f'Shapes: {[y.shape for y in ys]}')

    if normalize:
        ys = [normalize_samples(y) for y in ys]
    return ys
