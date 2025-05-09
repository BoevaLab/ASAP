from typing import Tuple, List

import numpy as np
import pyBigWig

from asap.dataloader.utils.data_bed import filter_idx_by_bed, filter_idx_by_unmap_threshold, whole_genome_idx
from asap.dataloader.utils.data_bw import get_binned_signal
from asap.dataloader.utils.seq import get_chr_seq
from asap.dataloader.utils.io import get_bw_from_file
from asap.dataloader.utils.fnv64 import hash_dn
from tqdm import tqdm


def get_wg_filtered_data(genome: str, signal_files: List[str], chrom: int, window_size: int, margin_size: int,
                         step_size: int, bin_size: int, blacklist_bed_files: List[str] = None,
                         unmappable_bed_file: str = None, unmap_threshold: float = None,
                         lower_bound: int = None, memmap=True, generated=None) -> Tuple[np.ndarray, np.ndarray]:
    idx = whole_genome_idx(genome=genome, chrom=chrom, step_window=step_size)

    return idx_to_filtered_data(genome=genome, signal_files=signal_files, seq_starts=idx, chrom=chrom, window_size=window_size,
                                margin_size=margin_size, bin_size=bin_size, blacklist_bed_files=blacklist_bed_files,
                                unmappable_bed_file=unmappable_bed_file, unmap_threshold=unmap_threshold, lower_bound=lower_bound,
                                memmap=memmap, generated=generated)


def idx_to_filtered_data(genome: str, signal_files: List[str], seq_starts: np.ndarray, chrom: int, window_size: int, margin_size: int,
                         bin_size: int, blacklist_bed_files: List[str] = None, unmappable_bed_file: str = None,
                         unmap_threshold: float = None, lower_bound: int = None, memmap=True, generated=None) -> Tuple[np.ndarray, np.ndarray]:
    mmap_mode = 'r' if memmap else None
    str_ = (f'chr={chrom}, window={window_size}, margin={margin_size}, bin={bin_size}, idx={seq_starts.shape}, '
          f'blacklist={blacklist_bed_files}, unmappable={unmappable_bed_file}, th={unmap_threshold}, '
          f'lower_bound={lower_bound}, files={signal_files}, include_map={True}...')
    id_ = hash_dn(str_, salt='0')
    try:
        print(f'Attempting to load data from file with {str_}')
        x = np.load(f'{generated}/{id_}_x.npy', mmap_mode=mmap_mode)
        y = np.load(f'{generated}/{id_}_y.npy', mmap_mode=mmap_mode)
        seq_starts = np.load(f'{generated}/{id_}_seq.npy', mmap_mode=mmap_mode) 
        print('\t...done!')
        
    except FileNotFoundError:
        print('\t...not found.')
        print('Generating data...')
        # Filter by blacklists
        if blacklist_bed_files is not None:
            for bl_bed in blacklist_bed_files:
                seq_starts = filter_idx_by_bed(chrom=chrom, seq_starts=seq_starts, window_size=window_size,
                                            blacklist_bed_file=bl_bed)


        # Filter out sequences extending beyond the chrom
        seq = get_chr_seq(genome, chrom)
        seq_starts = seq_starts[seq_starts + window_size + margin_size < len(seq)]
        seq_starts = seq_starts[seq_starts - margin_size > 0]

        # Filter by unmappable
        mappability = None
        if unmappable_bed_file is not None:
            if unmap_threshold == 0:
                seq_starts = filter_idx_by_bed(chrom=chrom, seq_starts=seq_starts, window_size=window_size + 2* margin_size,
                                            blacklist_bed_file=unmappable_bed_file)
            else:
                seq_starts, mappability = filter_idx_by_unmap_threshold(chrom=chrom, seq_starts=seq_starts, window_size=window_size + 2* margin_size,
                                                        unmappable_bed_file=unmappable_bed_file,
                                                        threshold=unmap_threshold, return_unmap=True)
        x, y = get_data_by_idx(genome, signal_files, chrom, seq_starts, window_size, margin_size, bin_size)

        # Filter by lower bound
        if lower_bound is not None:
            indices_over_bound = y.max(axis=(1, 2), initial=0) >= lower_bound
            x = x[indices_over_bound]
            y = y[indices_over_bound]
            seq_starts = seq_starts[indices_over_bound]
            if mappability is not None:
                mappability = mappability[indices_over_bound]
            print(f'Data after filtering bound {lower_bound}: x={x.shape} y={y.shape}')

        x = x.astype('int8')
        if mappability is None:
            mappability = np.ones_like(x)
        else:
            mappability = mappability.astype('int8')
        x = np.stack([x, mappability], axis=-1)
        y = y.astype('float32')
        
        # save and reload in mmap mode
        np.save(f'{generated}/{id_}_x.npy', x)
        np.save(f'{generated}/{id_}_y.npy', y)
        np.save(f'{generated}/{id_}_seq.npy', seq_starts)

        x = np.load(f'{generated}/{id_}_x.npy', mmap_mode=mmap_mode)
        y = np.load(f'{generated}/{id_}_y.npy', mmap_mode=mmap_mode)
        seq_starts = np.load(f'{generated}/{id_}_seq.npy', mmap_mode=mmap_mode)
        print('\t...done!')
    return x, y, seq_starts


def get_data_by_idx(genome: str, signal_files: List[str], chrom: int, seq_starts: np.ndarray, window: int, margin:int, bin_size: int) -> Tuple[
    np.ndarray, np.ndarray]:
    seq = get_chr_seq(genome, chrom)
    x = _get_x_by_idx(chrom=chrom, seq=seq, seq_starts=seq_starts, window=window, margin=margin)
    y = _get_y_by_idx(signal_files, chrom, seq_starts, window, bin_size)
    return x, y


def _get_binned_whole_signals(signal_files: List[str], chrom: int, bin_size: int, start: int, end: int):
    whole_signals = []
    for signal_file in signal_files:
        signal = get_binned_signal(signal_file, chrom, start=start, end=end, bin_size=bin_size)
        whole_signals.append(signal)
    whole_signals = np.column_stack(whole_signals)  # shape (signal_nr_bins, len(signal_files))
    return whole_signals


def _get_y_by_idx(signal_files: List[str], chrom: int, seq_starts: np.ndarray, window: int,
                  bin_size: int) -> np.ndarray:
    print(f'Getting y with index ({seq_starts.shape})')
    start, end = seq_starts[0], seq_starts[-1] + window
    nr_bins = window // bin_size

    whole_signals = _get_binned_whole_signals(signal_files, chrom, bin_size=1, start=start, end=end)

    window_idx = seq_starts - start
    idx = window_idx[:, np.newaxis] + np.arange(window)  # Generate indices for slicing whole_signals

    indexed_windows = whole_signals[idx, :].reshape((len(idx), nr_bins, bin_size, len(signal_files)))
    y = indexed_windows.max(axis=2)

    print(f'Got y with index ({seq_starts.shape}): {y.shape}')
    return y


def _get_x_by_idx(chrom: int, seq: np.ndarray, seq_starts: np.ndarray, window: int, margin: int) -> np.ndarray:
    print(f'Getting x with index ({seq_starts.shape})')
    x_starts = seq_starts[:, np.newaxis] + np.arange(window + 2*margin) - margin
    indexed_seq = seq[x_starts]
    print(f'Got x with index ({seq_starts.shape}): {indexed_seq.shape}')
    return indexed_seq

def write_predictions_to_bigwig(file_name: str, preds: np.ndarray, chroms, seq_starts: np.ndarray, genome: str, bin_size) -> None:
    assert pyBigWig.numpy == 1, 'pyBigWig compiled without numpy support!'
    print("Opening bigwig file...")
    bw = get_bw_from_file(file_name, file_mode='w')

    print("Generating bigwig header...")
    header = [('chr' + str(chrom), len(get_chr_seq(genome, chrom))) for chrom in chroms]
    print(header)
    bw.addHeader(header)

    for i, chrom in enumerate(tqdm(chroms)):
        print(f"Exporting data for chromosome {chrom}")
        start = seq_starts[i]
        for j in range(preds[i].shape[0]):
            pred_len = preds[i].shape[1]
            starts = start[j] + np.arange(pred_len)* bin_size
            bw.addEntries('chr' + str(chrom), starts, values=preds[i][j], span=bin_size)
    bw.close()
