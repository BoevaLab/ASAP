import pandas as pd
import os
import sys


def vcf_to_bed(vcf_filename):
    if vcf_filename.endswith('.vcf.gz'):
        bed_filename = vcf_filename[:-7] + '.bed'
    else:
        bed_filename = vcf_filename[:-4] + '.bed'

    compression = 'gzip' if vcf_filename.endswith('.gz') else None
    vcf_data = pd.read_csv(vcf_filename, comment='#', sep='\t', header=None,
                           names=['chrom', 'pos', 'id', 'ref', 'alt', 'qual', 'filter', 'info'],
                           compression=compression)
    bed_data = vcf_data[['chrom', 'pos']].copy()
    bed_data['end'] = bed_data['pos'] + 1

    print('Writing BED file to:', bed_filename)
    bed_data.to_csv(bed_filename, sep='\t', header=False, index=False)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python script.py <input.vcf or input.vcf.gz>')
        sys.exit(1)

    vcf_file = sys.argv[1]
    vcf_to_bed(vcf_file)
