import pandas as pd

def make_pcawg_df(snv_file: str):
    vcf_columns = ['chr', 'pos', 'id', 'ref', 'alt', 'qual', 'filter', 'info']
    df = pd.read_csv(snv_file, comment='#', names=vcf_columns, sep='\t', index_col=False)
    df = df.sort_values(by=['chr', 'pos'])
    df[df.chr.astype(str).str.replace('chr', '').astype(int).isin(range(1, 23))] # TODO test
    df = df[df['filter'] == 'PASS']
    
    output_columns = ['id', 'chr', 'pos', 'ref', 'alt']
    return df[output_columns]
