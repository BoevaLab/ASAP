import numpy as np
import pandas as pd
from tqdm import tqdm


def filter_snv(df: pd.DataFrame):
    print('Original df rows:', len(df))
    df = df[df['filter'] == 'PASS']
    print('df rows after filtering by "PASS":', len(df))

    # Filter based on nearby mutations in window
    window = 200
    print('Filtering by overlapping window')
    df = _filter_by_overlapping_window(df, window=window)

    # Filter by read depth for fold change to make sense
    df = df[(df.atac_vaf > 0) & (df.atac_vaf < 1)]
    print('df rows after filtering by 0 < ATAC_VAF < 1:', len(df))
    df['atac_reads'] = df.atac_ref_reads + df.atac_alt_reads
    df = df[df.atac_reads >= 10]
    print('df rows after filtering by atac_reads >= 10', len(df))

    # Filter out blacklisted and unmappable
    print('Filtering by blacklisted and unmappable')
    df = _filter_by_blacklist(df)
    return df


def _filter_by_overlapping_window(df, window=2000):
    df = df.copy()
    df_sorted = df.sort_values(['chr', 'pos'])

    mask = ~(
            ((df_sorted['chr'] == df_sorted['chr'].shift(1))
             & (df_sorted['pos'] - df_sorted['pos'].shift(1) <= window))
            | ((df_sorted['chr'] == df_sorted['chr'].shift(-1))
               & (df_sorted['pos'].shift(-1) - df_sorted['pos'] <= window))
    )
    filtered_df = df_sorted[mask]
    filtered_df = filtered_df.sort_index()
    print(f'Original {len(df)} samples. Filtered to {len(filtered_df)} samples.')
    return filtered_df


def _filter_by_blacklist(df):
    filtered_df = df.copy()

    def _is_not_blacklisted(row, blacklist):
        chr_blacklist = blacklist[blacklist['chr'] == row['chr']]
        overlaps = chr_blacklist.apply(lambda x: x['start'] <= row['pos'] <= x['end'], axis=1)
        return not overlaps.any()

    bl_files = [
        '/Users/liine/PycharmProjects/OpenChromatinPrediction/data/basenji_blacklist.bed',
    ]
    for file in bl_files:
        blacklist_df = pd.read_csv(file, sep='\t', header=None, names=['chr', 'start', 'end'])
        filtered_df = filtered_df[filtered_df.apply(_is_not_blacklisted, axis=1, blacklist=blacklist_df)]
    print(f'Original {len(df)} samples. Filtered to {len(filtered_df)} samples.')
    return filtered_df


def add_unmap(df, window=1000):
    side = window // 2
    unmappable_bed_file = '/Users/liine/PycharmProjects/OpenChromatinPrediction/data/basenji_unmappable.bed'
    unmap = pd.read_csv(unmappable_bed_file, delimiter='\t', header=None, names=['chr', 'start', 'end'])

    for i, row in tqdm(df.iterrows(), total=len(df)):
        chrom, start, end = row['chr'], row.pos - side, row.pos + side

        chr_unmap = unmap[unmap['chr'] == chrom]

        overlaps = chr_unmap[(start < chr_unmap.start) & (chr_unmap.start < end)
                             | (start < chr_unmap.end) & (chr_unmap.end < end)
                             | (chr_unmap.start < start) & (end < chr_unmap.end)]
        starts = overlaps.start.to_numpy()
        ends = overlaps.end.to_numpy()
        starts[starts < start] = start
        ends[ends > end] = end

        overlap_length = sum(overlaps.end - overlaps.start)
        df.loc[i, ['unmap']] = overlap_length / window
    return df
