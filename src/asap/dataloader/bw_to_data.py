from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import pyBigWig

from asap.dataloader.utils.data_bed import filter_idx_by_bed, whole_genome_idx
from asap.dataloader.utils.data_bw import get_binned_signal
from asap.dataloader.utils.seq import get_chr_seq
from asap.dataloader.utils.io import get_bw_from_file
from asap.dataloader.utils.fnv64 import hash_dn
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Whole-chromosome caches (sequence, mappability, signal).
#
# Unlike the per-window caches these used to be, these depend only on
# (genome, chrom[, signal_files/unmap_file]) -- not on window_size, margin_size,
# step_size or bin_size. They're generated once per chromosome and reused
# across every window/step/margin configuration and every epoch.
# ---------------------------------------------------------------------------

def get_cached_chrom_seq(genome: str, chrom: int, generated: str, memmap: bool = True) -> np.ndarray:
    mmap_mode = 'r' if memmap else None
    id_ = hash_dn(f'chromseq: genome={genome}, chrom={chrom}', salt='0')
    path = f'{generated}/{id_}_chromseq.npy'
    try:
        return np.load(path, mmap_mode=mmap_mode)
    except FileNotFoundError:
        seq = get_chr_seq(genome, chrom).astype('int8')
        np.save(path, seq)
        return np.load(path, mmap_mode=mmap_mode)


def get_cached_chrom_mappability(genome: str, chrom: int, unmappable_bed_file: Optional[str],
                                 chrom_len: int, generated: str, memmap: bool = True) -> Optional[np.ndarray]:
    if unmappable_bed_file is None:
        return None
    mmap_mode = 'r' if memmap else None
    id_ = hash_dn(f'chrommap: genome={genome}, chrom={chrom}, unmap={unmappable_bed_file}', salt='0')
    path = f'{generated}/{id_}_chrommap.npy'
    try:
        return np.load(path, mmap_mode=mmap_mode)
    except FileNotFoundError:
        track = np.ones(chrom_len, dtype='int8')
        unmap = pd.read_csv(unmappable_bed_file, delimiter='\t', header=None, names=['chr', 'start', 'end'])
        unmap = unmap[(unmap.chr == f'chr{chrom}') | (unmap.chr.astype(str) == f'{chrom}')]
        for gap_start, gap_end in zip(unmap.start, unmap.end):
            s = max(0, int(gap_start))
            e = min(chrom_len, int(gap_end))
            if e > s:
                track[s:e] = 0
        np.save(path, track)
        return np.load(path, mmap_mode=mmap_mode)


def get_cached_chrom_signal(genome: str, chrom: int, signal_files: Optional[List[str]], chrom_len: int,
                            generated: str, memmap: bool = True) -> Optional[np.ndarray]:
    if signal_files is None:
        return None
    mmap_mode = 'r' if memmap else None
    id_ = hash_dn(f'chromsignal: genome={genome}, chrom={chrom}, files={signal_files}', salt='0')
    path = f'{generated}/{id_}_chromsignal.npy'
    try:
        return np.load(path, mmap_mode=mmap_mode)
    except FileNotFoundError:
        whole_signals = _get_binned_whole_signals(signal_files, chrom, bin_size=1, start=0, end=chrom_len)
        y = whole_signals.astype('float32')
        np.save(path, y)
        return np.load(path, mmap_mode=mmap_mode)


def get_cached_chrom_data(genome: str, chrom: int, signal_files: Optional[List[str]],
                          unmappable_bed_file: Optional[str], generated: str,
                          memmap: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    chrom_seq = get_cached_chrom_seq(genome, chrom, generated, memmap=memmap)
    chrom_len = len(chrom_seq)
    mappability = get_cached_chrom_mappability(genome, chrom, unmappable_bed_file, chrom_len, generated, memmap=memmap)
    chrom_y = get_cached_chrom_signal(genome, chrom, signal_files, chrom_len, generated, memmap=memmap)
    return chrom_seq, mappability, chrom_y


# ---------------------------------------------------------------------------
# Index building: which positions are valid samples, given filtering rules.
# This is the only piece that still depends on window_size/margin_size/
# step_size/bin_size/blacklist/unmap_threshold -- and it's cheap (an array of
# positions), so it's cached separately from the (much larger) chromosome data.
# ---------------------------------------------------------------------------

def _unmappable_fraction(mappability_track: np.ndarray, starts: np.ndarray, extent: int) -> np.ndarray:
    # Fraction of unmappable bases in [start, start + extent) for each start.
    # NOTE: mirrors the original filter_idx_by_unmap_threshold behaviour exactly,
    # which checks [start, start + extent) -- i.e. anchored at `start`, not
    # symmetric around it -- even though the data actually read is centered
    # (start - margin, start + window + margin). Preserved intentionally rather
    # than "fixed", since changing it would silently change which windows are
    # kept.
    unmap = (1 - mappability_track).astype(np.int64)
    prefix = np.concatenate(([0], np.cumsum(unmap)))
    counts = prefix[starts + extent] - prefix[starts]
    return counts / extent


def build_seq_starts_index(genome: str, chrom: int, seq_starts: np.ndarray, window_size: int, margin_size: int,
                           chrom_len: int, mappability_track: Optional[np.ndarray], chrom_y: Optional[np.ndarray],
                           bin_size: int, blacklist_bed_files: List[str] = None, unmappable_bed_file: str = None,
                           unmap_threshold: float = None, lower_bound: int = None, generated: str = None) -> np.ndarray:
    fingerprint = f'{len(seq_starts)}:{int(seq_starts[0]) if len(seq_starts) else 0}:' \
                  f'{int(seq_starts[-1]) if len(seq_starts) else 0}:{int(seq_starts.sum())}'
    str_ = (f'idx: chr={chrom}, window={window_size}, margin={margin_size}, bin={bin_size}, '
            f'candidates={fingerprint}, blacklist={blacklist_bed_files}, unmappable={unmappable_bed_file}, '
            f'unmap_th={unmap_threshold}, lower_bound={lower_bound}, genome={genome}')
    id_ = hash_dn(str_, salt='0')
    path = f'{generated}/{id_}_idx.npy'
    try:
        print(f'Attempting to load index from file with {str_}')
        seq_starts = np.load(path)
        print('\t...done!')
        return seq_starts
    except FileNotFoundError:
        print('\t...not found.')
        print('Generating index...')

    # Filter by blacklists
    if blacklist_bed_files is not None:
        for bl_bed in blacklist_bed_files:
            seq_starts = filter_idx_by_bed(chrom=chrom, seq_starts=seq_starts, window_size=window_size,
                                        blacklist_bed_file=bl_bed)

    # Filter out sequences extending beyond the chrom
    seq_starts = seq_starts[seq_starts + window_size + margin_size < chrom_len]
    seq_starts = seq_starts[seq_starts - margin_size > 0]

    # Filter by unmappable, using the whole-chromosome mappability track.
    # Extent matches the original: [seq_starts, seq_starts + window_size + 2*margin_size).
    if mappability_track is not None and unmap_threshold is not None:
        extent = window_size + 2 * margin_size
        if unmap_threshold == 0:
            # Strict mode: reuse filter_idx_by_bed directly against the unmap bed
            # file (exactly as the original did), rather than deriving "zero
            # overlap" from the mappability track -- the two have different
            # (inclusive vs. half-open) boundary handling at exact edges, and
            # this keeps strict-mode behavior bit-identical to before.
            seq_starts = filter_idx_by_bed(chrom=chrom, seq_starts=seq_starts, window_size=extent,
                                        blacklist_bed_file=unmappable_bed_file)
        else:
            # --- Correct behavior (current) ---
            # Fraction of bases in [start, start+extent) that are actually
            # unmappable, computed once for the whole chromosome via a prefix
            # sum over the mappability track, then sliced per candidate window.
            frac = _unmappable_fraction(mappability_track, seq_starts, extent)
            seq_starts = seq_starts[frac <= unmap_threshold]

            # --- Old (buggy) behavior, kept here for reference ---
            # The original filter_idx_by_unmap_threshold (data_bed.py) computed,
            # per window, `overlap_length = sum(overlaps.end - overlaps.start)`
            # using the RAW, unclamped length of every unmap interval that
            # touched the window at all -- even when only a sliver of that
            # interval actually fell inside the window. So a single long
            # unmappable region (e.g. 60bp) could cause a window that only
            # overlapped it by a handful of bp to be dropped entirely, because
            # the full 60bp counted against the threshold budget instead of the
            # true (much smaller) overlap. This was a bug, not an intentional
            # design choice -- verified against the original by differential
            # testing (see conversation/PR notes). If bit-identical window
            # selection against an old cached index or trained checkpoint is
            # ever needed, this reproduces the dominant effect of that bug
            # (raw interval length, not clipped overlap) without reintroducing
            # the original's slow per-window pandas loop -- it loops over
            # unmap intervals instead, which are typically far fewer than
            # candidate windows. Uncomment in place of the two lines above:
            #
            # unmap_df = pd.read_csv(unmappable_bed_file, delimiter='\t', header=None,
            #                        names=['chr', 'start', 'end'])
            # unmap_df = unmap_df[(unmap_df.chr == f'chr{chrom}') | (unmap_df.chr.astype(str) == f'{chrom}')]
            # window_end = seq_starts + extent
            # raw_overlap_length = np.zeros(len(seq_starts), dtype=np.int64)
            # for gs, ge in zip(unmap_df.start.to_numpy(), unmap_df.end.to_numpy()):
            #     touches = (seq_starts < ge) & (gs < window_end)
            #     raw_overlap_length[touches] += (ge - gs)  # bug: full interval length, not clipped
            # seq_starts = seq_starts[raw_overlap_length <= unmap_threshold * extent]

    # Filter by lower bound
    if lower_bound is not None and chrom_y is not None:
        idx = seq_starts[:, np.newaxis] + np.arange(window_size)
        max_signal = chrom_y[idx].max(axis=(1, 2))
        seq_starts = seq_starts[max_signal >= lower_bound]

    print(f'Generated index: {seq_starts.shape}')
    np.save(path, seq_starts)
    return seq_starts


def get_wg_filtered_data(genome: str, signal_files: List[str], chrom: int, window_size: int, margin_size: int,
                         step_size: int, bin_size: int, blacklist_bed_files: List[str] = None,
                         unmappable_bed_file: str = None, unmap_threshold: float = None,
                         lower_bound: int = None, memmap=True, generated=None):
    idx = whole_genome_idx(genome=genome, chrom=chrom, step_window=step_size)

    return idx_to_filtered_data(genome=genome, signal_files=signal_files, seq_starts=idx, chrom=chrom, window_size=window_size,
                                margin_size=margin_size, bin_size=bin_size, blacklist_bed_files=blacklist_bed_files,
                                unmappable_bed_file=unmappable_bed_file, unmap_threshold=unmap_threshold, lower_bound=lower_bound,
                                memmap=memmap, generated=generated)


def idx_to_filtered_data(genome: str, signal_files: List[str], seq_starts: np.ndarray, chrom: int, window_size: int, margin_size: int,
                         bin_size: int, blacklist_bed_files: List[str] = None, unmappable_bed_file: str = None,
                         unmap_threshold: float = None, lower_bound: int = None, memmap=True, generated=None):
    chrom_seq, mappability, chrom_y = get_cached_chrom_data(
        genome=genome, chrom=chrom, signal_files=signal_files, unmappable_bed_file=unmappable_bed_file,
        generated=generated, memmap=memmap,
    )
    seq_starts = build_seq_starts_index(
        genome=genome, chrom=chrom, seq_starts=seq_starts, window_size=window_size, margin_size=margin_size,
        chrom_len=len(chrom_seq), mappability_track=mappability, chrom_y=chrom_y, bin_size=bin_size,
        blacklist_bed_files=blacklist_bed_files, unmappable_bed_file=unmappable_bed_file,
        unmap_threshold=unmap_threshold, lower_bound=lower_bound, generated=generated,
    )
    return chrom_seq, mappability, chrom_y, seq_starts


# ---------------------------------------------------------------------------
# Unchanged below: ad hoc per-position window extraction (used by snv/predict.py
# for one-off SNV lookups, not by the cached bulk dataset path above) and
# bigwig export.
# ---------------------------------------------------------------------------

def get_data_by_idx(genome: str, signal_files: List[str], chrom: int, seq_starts: np.ndarray, window: int, margin:int, bin_size: int) -> Tuple[
    np.ndarray, np.ndarray]:
    seq = get_chr_seq(genome, chrom)
    x = _get_x_by_idx(chrom=chrom, seq=seq, seq_starts=seq_starts, window=window, margin=margin)
    y = None
    if signal_files is not None:
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
