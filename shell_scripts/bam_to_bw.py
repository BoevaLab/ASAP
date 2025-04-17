import argparse
import os
import signal
import subprocess
import time

PICARD_PATH = None


def atac_filter(raw_bam: str, chr_sizes: str):
    # NB! paired-end unfiltered BAM
    prefix = '.'.join(os.path.basename(raw_bam).split('.')[:-1])

    # filter reads by quality, paired-end
    cmd = f'samtools view -F 1804 -f 2 -q 30 -u {raw_bam} | ' \
          f'samtools sort -n /dev/stdin -o {prefix}.tmp_filt.bam -T {prefix}'
    run_shell_cmd(cmd=cmd)

    # correct paired reads
    run_shell_cmd(cmd=f'samtools fixmate -r {prefix}.tmp_filt.bam {prefix}.fixmate.bam')
    run_shell_cmd(cmd=f'rm -f {prefix}.tmp_filt.bam')

    # filter corrected reads bam
    cmd = f'samtools view -F 1804 -f 2 -u {prefix}.fixmate.bam | ' \
          f'samtools sort /dev/stdin -o {prefix}.filt.bam -T {prefix}'
    run_shell_cmd(cmd=cmd)
    run_shell_cmd(cmd=f'rm -f {prefix}.fixmate.bam')

    assert get_bam_lines(f'{prefix}.filt.bam') > 0, 'No reads found after filtering "samtools fixmate"d PE BAM'

    # mark duplicates
    cmd = f'java -jar {PICARD_PATH} MarkDuplicates ' \
          f'INPUT={prefix}.filt.bam OUTPUT={prefix}.dupmark.bam ' \
          f'VALIDATION_STRINGENCY=LENIENT ASSUME_SORTED=TRUE REMOVE_DUPLICATES=FALSE ' \
          f'USE_JDK_DEFLATER=TRUE USE_JDK_INFLATER=TRUE'
    # f'METRICS_FILE={prefix}.dup.qc '
    run_shell_cmd(cmd=cmd)

    # remove duplicates
    run_shell_cmd(cmd=f'samtools view -F 1804 -f 2 -b {prefix}.dupmark.bam  > {prefix}.nodup.bam')

    # filter out mitochondrial reads
    run_shell_cmd(cmd=f'zcat -f {chr_sizes} | ' \
                      "grep -v -P '^(chrM|MT)\s' | " \
                      "awk 'BEGIN{OFS=\"\t\"} {print $1,0,$2}' > " \
                      f'{prefix}.nodup.no_chrM_MT.tmp.chrsz')
    run_shell_cmd(cmd=f'samtools view -b -L {prefix}.nodup.no_chrM_MT.tmp.chrsz {prefix}.nodup.bam > '
                      f'{prefix}.nodup.no_chrM_MT.bam')
    run_shell_cmd(cmd=f'samtools index {prefix}.nodup.no_chrM_MT.bam')
    run_shell_cmd(cmd=f'rm -f {prefix}.nodup.no_chrM_MT.tmp.chrsz')

    assert get_bam_lines(f'{prefix}.nodup.no_chrM_MT.bam') > 0, 'No reads in final filtered bam!'


def main():
    args = parse()
    assert args.bam_file.endswith('.bam')

    bam_file = filter_bam(bam_file=args.bam_file, out_prefix=args.out_prefix)
    macs2_signal_track(bam_file=bam_file, prefix=args.out_prefix, chr_sz=args.chrom_sizes)


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog='python bam_to_bw.py')

    parser.add_argument('-f', '--bam_file', type=str, required=True)
    parser.add_argument('-o', '--out_prefix', type=str, required=True)
    parser.add_argument('--chrom_sizes', type=str, required=True)

    return parser.parse_args()


def assert_tools():
    pass


def filter_bam(bam_file, out_prefix):
    sorted_file = out_prefix + '.sorted.bam'
    filtered_file = out_prefix + '.filtered.bam'
    final_file = out_prefix + '.final.bam'

    # run_shell_cmd(f'samtools sort -o {sorted_file} {bam_file}')
    # run_shell_cmd(f'samtools index {sorted_file}')
    #
    # # filter by MAPQ quality >= 10
    # run_shell_cmd(f'samtools view -f 2 -q 10 -b -@ 20 {sorted_file} > {filtered_file}')
    # run_shell_cmd(f'samtools index {filtered_file}')
    #
    # # remove duplicates
    # run_shell_cmd(
    #     f'java -jar {PICARD_PATH} MarkDuplicates --INPUT {filtered_file} --OUTPUT {final_file} '
    #     f'--METRICS_FILE marked_dup_metrics.txt --VALIDATION_STRINGENCY LENIENT --REMOVE_DUPLICATES TRUE')
    return final_file


def macs2_signal_track(bam_file, prefix, chr_sz, gen_sz='hs', pval_thresh=0.01, smooth_win=150):
    # gen_sz - Genome size (sum of entries in 2nd column of chr. sizes file, or hs for human, ms for mouse)
    fc_bigwig = f'{prefix}.fc.signal.bigwig'
    pval_bigwig = f'{prefix}.pval.signal.bigwig'

    # temporary files
    pval_bedgraph = f'{prefix}.pval.signal.bedgraph'
    pval_bedgraph_srt = f'{prefix}.pval.signal.srt.bedgraph'

    shift_size = -int(round(float(smooth_win) / 2.0))
    temp_files = []

    run_shell_cmd(
        f'macs2 callpeak -t {bam_file} -f BAM -n {prefix} -g {gen_sz} -p {pval_thresh} '
        f'--shift {shift_size} --extsize {smooth_win} '
        '--nomodel -B --SPMR --keep-dup all --call-summits '
    )
    #
    # # sval counts the number of tags per million in the (compressed) BAM file
    # sval = get_bam_lines(bam_file) / 1000000.0
    #
    # run_shell_cmd(
    #     f'macs2 bdgcmp -t "{prefix}_treat_pileup.bdg" -c "{prefix}_control_lambda.bdg" '
    #     f'--o-prefix {prefix} -m ppois -S {sval}')

    # extra step: remove chromosomes not listed from the bedgraph
    # run_shell_cmd("awk 'NR==FNR{chromosomes[$1]; next} $1 in chromosomes' " +
    #               f'{chr_sz} {prefix}_ppois.bdg > {prefix}_ppois2.bdg')
    #
    # run_shell_cmd(f'bedtools slop -i "{prefix}_ppois2.bdg" -g {chr_sz} -b 0 | bedClip stdin {chr_sz} {pval_bedgraph}')
    #
    # # sort and remove any overlapping regions in bedgraph by comparing two lines in a row
    # sort_fn = 'gsort'  # 'sort'
    # run_shell_cmd(
    #     f'LC_COLLATE=C {sort_fn} -k1,1 -k2,2n {pval_bedgraph} | '
    #     f'awk \'BEGIN{{OFS="\\t"}}{{if (NR==1 || NR>1 && (prev_chr!=$1 '
    #     f'|| prev_chr==$1 && prev_chr_e<=$2)) '
    #     f'{{print $0}}; prev_chr=$1; prev_chr_e=$3;}}\' > {pval_bedgraph_srt}'
    # )
    # rm_f(pval_bedgraph)
    #
    # run_shell_cmd(f'bedGraphToBigWig {pval_bedgraph_srt} {chr_sz} {pval_bigwig}')
    # rm_f(pval_bedgraph_srt)

    # remove temporary files
    # temp_files.append(f'{prefix}_*')
    # rm_f(temp_files)

    return fc_bigwig, pval_bigwig


def get_bam_lines(bam_file) -> int:
    return int(run_shell_cmd(f'samtools view -c {bam_file}'))


def get_ticks():
    """Returns ticks.
        - Python3: Use time.perf_counter().
        - Python2: Use time.time().
    """
    return getattr(time, 'perf_counter', getattr(time, 'time'))()


def run_shell_cmd(cmd):
    p = subprocess.Popen(
        ['/bin/bash', '-o', 'pipefail'],  # to catch error in pipe
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        preexec_fn=os.setsid)  # to make a new process with a new PGID
    pid = p.pid
    pgid = os.getpgid(pid)
    print('run_shell_cmd: PID={}, PGID={}, CMD={}'.format(pid, pgid, cmd))
    t0 = get_ticks()
    stdout, stderr = p.communicate(cmd)
    rc = p.returncode
    t1 = get_ticks()
    err_str = f'PID={pid}, PGID={pgid}, RC={rc}, DURATION_SEC={t1 - t0:.1f}\n' \
              f'STDERR={stderr.strip()}\nSTDOUT={stdout.strip()}'

    if rc:
        # kill all child processes
        try:
            os.killpg(pgid, signal.SIGKILL)
        except:
            pass
        finally:
            raise Exception(err_str)
    else:
        print(err_str)
    return stdout.strip('\n')


def rm_f(files):
    if files:
        if type(files) == list:
            run_shell_cmd(f'rm -f {" ".join(files)}')
        else:
            run_shell_cmd(f'rm -f {files}')


if __name__ == '__main__':
    PICARD_PATH = run_shell_cmd('which picard.jar')
    assert os.path.exists(PICARD_PATH), 'picard.jar not in path!'
    main()
