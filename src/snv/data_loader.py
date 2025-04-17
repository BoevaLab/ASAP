import os

import pandas as pd
import pysam

from src.config import CellLineConfig


def get_pcawg_df(cell_line: CellLineConfig, output_path):
    if os.path.exists(output_path):
        print(f'Loading SNV predictions from: {os.path.abspath(output_path)}')
        df = pd.read_csv(output_path, index_col=False)
    else:
        print('Making new SNV prediction file')
        df = make_pcawg_df(cell_line.snv)
        #add_atac_reads(df, cell_line.bam)
    return df


def make_pcawg_df(snv_file: str):
    output_columns = ['chr', 'pos', 'ref', 'alt', 'confidence', 'pval', 'odds_ratio', 'atac_ref_reads', 'atac_alt_reads', 'wgs_vaf']
    df = pd.read_csv(snv_file)
    df = df.sort_values(by=['chr', 'pos'])
    df = df[df.chr.isin([f'chr{chrom}' for chrom in range(1, 23)])]
    df['alt'] = df['var']
    df['atac_alt_reads'] = df['atac_var_reads']
    return df[output_columns]


def add_atac_reads(df, atac_bam_file: str):
    df['atac_ref_reads'] = 0
    df['atac_alt_reads'] = 0
    df['atac_vaf'] = 0

    # Add ATAC for ref and alt from bam file
    bam = pysam.AlignmentFile(atac_bam_file, 'r')
    pysam_order = ['A', 'C', 'G', 'T']

    for i, row in df.iterrows():
        coverage = bam.count_coverage(
            contig=row.chr,
            start=row.pos - 1,
            stop=row.pos,
        )
        cov = dict({c: coverage[i][0] for i, c in enumerate(pysam_order)})
        r, a = cov[row.ref], cov[row.alt]
        df.loc[i, 'atac_ref_reads'] = r
        df.loc[i, 'atac_alt_reads'] = a
        df.loc[i, 'atac_vaf'] = a / (r + a) if (r + a) != 0 else None
