import numpy as np
import pandas as pd
from tqdm import tqdm
from .seq import get_range_by_chrom_number
from typing import Tuple
import gzip

def whole_genome_idx(genome:str , chrom: int, step_window: int) -> np.ndarray:
    start, end = get_range_by_chrom_number(genome, chrom)
    seq_starts = np.arange(start, end, step_window)
    return seq_starts

def load_vcf_file(vcf_path: str) -> pd.DataFrame:
    with open(vcf_path, 'r') if not vcf_path.endswith('.gz') else gzip.open(vcf_path, 'rt') as f:
        # Find the column header line (starts with #CHROM)
        for line in f:
            if line.startswith("#CHROM"):
                header = line.strip().lstrip("#").split('\t')
                break
        
        df = pd.read_csv(f, sep='\t', names=header, comment='#')
        df['start'] = df['POS'] - 1
        df['end'] = df['POS'] + df['REF'].str.len() - 2
        
        df['chr'] = df['CHROM'].str.replace('chr', '', regex=False)
        return df[['chr', 'start', 'end']]

def filter_idx_by_bed(chrom: int, seq_starts: np.ndarray, window_size: int, blacklist_bed_file: str) -> np.ndarray:
    print(f'Filtering chr{chrom} {len(seq_starts)} samples with blacklist file: {blacklist_bed_file}')
    nr_samples = len(seq_starts)
    if blacklist_bed_file.endswith(".vcf.gz") or blacklist_bed_file.endswith(".vcf"):
        bed = load_vcf_file(blacklist_bed_file)
    else:
        bed = pd.read_csv(blacklist_bed_file, delimiter='\t', header=None, names=['chr', 'start', 'end'])
    bed = bed[bed.chr == f'chr{chrom}']
    for gap_start, gap_end in zip(bed.start, bed.end):
        seq_starts = seq_starts[(gap_start > seq_starts + window_size) | (gap_end < seq_starts)]
    print(f'Filtered out {nr_samples - len(seq_starts)} samples. New size: {len(seq_starts)}')
    return seq_starts


def construct_indices(start, end):
    # via https://stackoverflow.com/a/4708737
    lens = end - start
    np.cumsum(lens, out=lens)
    i = np.ones(lens[-1], dtype=int)
    i[0] = start[0]
    i[lens[:-1]] += start[1:]
    i[lens[:-1]] -= end[:-1]
    np.cumsum(i, out=i)
    return i


def filter_idx_by_unmap_threshold(chrom: int, seq_starts: np.ndarray, window_size: int, unmappable_bed_file: str,
                                  threshold: float = 0.35, return_unmap=False) -> Tuple[np.ndarray, np.ndarray]:
    print(f'Filtering chr{chrom} {len(seq_starts)} samples '
          f'with unmappable file (th={threshold}): {unmappable_bed_file}')
    if return_unmap:
        print('Generating mappability file')
    nr_samples = len(seq_starts)
    unmap = pd.read_csv(unmappable_bed_file, delimiter='\t', header=None, names=['chr', 'start', 'end'])
    unmap = unmap[unmap.chr == f'chr{chrom}']

    filtered = np.ones_like(seq_starts)
    if return_unmap:
        m = []

    for i, start in tqdm(enumerate(seq_starts), total=len(seq_starts)):
        end = start + window_size
        overlaps = unmap[(start < unmap.start) & (unmap.start < end)
                         | (start < unmap.end) & (unmap.end < end)
                         | (unmap.start < start) & (end < unmap.end)]
        starts = overlaps.start.to_numpy()
        ends = overlaps.end.to_numpy()
        starts[starts < start] = start
        ends[ends > start + window_size] = start + window_size
        overlap_length = sum(overlaps.end - overlaps.start)

        if threshold is not None and overlap_length > threshold * window_size:
            filtered[i] = 0
        elif return_unmap: # make mappability sequence
            mappable_ = np.ones(window_size)
            if len(starts) != 0:
                ind = construct_indices(starts-start, ends - start)
                mappable_[ind] = 0.
            m.append(mappable_)
    
    seq_starts = seq_starts[filtered == 1]
    print(f'Filtered out {nr_samples - len(seq_starts)} samples. New size: {len(seq_starts)}')
    if return_unmap:
        assert len(seq_starts) == len(m)
        return seq_starts, np.vstack(m)
    return seq_starts
