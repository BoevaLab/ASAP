import pandas as pd

def make_pcawg_df(snv_file: str):
    vcf_columns = ['chr', 'pos', 'id', 'ref', 'alt', 'qual', 'filter', 'info']
    df = pd.read_csv(snv_file, comment='#', names=vcf_columns, sep='\t', index_col=False)
    df = df.sort_values(by=['chr', 'pos'])
    df = df[df.chr.isin([f'chr{chrom}' for chrom in range(1, 23)])]
    df = df[df['filter'] == 'PASS']
    
    output_columns = ['id', 'chr', 'pos', 'ref', 'alt']
    return df[output_columns]
