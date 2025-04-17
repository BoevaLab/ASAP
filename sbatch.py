import argparse
import subprocess as sps

# EXAMPLES
# python sbatch.py --gpu 4 -t 24:00:00 --tmp 15 -p run_script=train train.fold=0 experiment_name=tcga +cell_line=TCGA-A6-A567
# python sbatch.py -p run_script=predict_snv model=dcnn experiment_name=tcga  +cell_line=TCGA-A6-A567

EXCLUDE = []
# we want nodes with scratch for tmp storage of datasets
NO_SCRATCH_NODES = ['01', '14', '15', '18', '19', '20', '21', '24', '25']

def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog='python run.py')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-c', '--cpu', type=int, default=1)
    parser.add_argument('-m', '--mem', type=int, default=32)
    parser.add_argument('--tmp', type=int, default=0)
    parser.add_argument('-t', '--time', type=str, default='12:00:00')
    parser.add_argument('-p', '--params', type=list, nargs='+', default=[])  # all hydra params
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()

    is_gpu = args.gpu > 0
    params = [''.join(param) for param in args.params]
    print('params:', params)
    job_name = '_'.join(params)

    partition = 'gpu' if is_gpu else 'compute,gpu'
    sub = ['sbatch', '--ntasks=1', f'--cpus-per-task={args.cpu}',
           f'--job-name={job_name}',
           f'--out={job_name}.txt',
           f'--mem={args.mem}G',
           f'--time={args.time}',
           '--partition', partition,
           ]
    if is_gpu:
        if 'xlstm' in job_name:
            sub += [f'--gres=gpu:rtx4090:{args.gpu}']
        else:
            sub += [f'--gres=gpu:rtx2080ti:{args.gpu}']
    if args.tmp:
        sub += ['--tmp', f'{args.tmp}G', ]
        EXCLUDE += NO_SCRATCH_NODES
    if EXCLUDE:
        sub += ['--exclude', ','.join(list(map(lambda node: 'gpu-biomed-' + node, EXCLUDE)))]

    env = 'gpu' if is_gpu else 'cpu'
    cmd = f'. ~/.bashrc; conda activate convnext-atac; python --version; which python; ' \
          f'python3 -c \'' \
          f'import torch; ' \
          f'print(torch.__version__, torch.cuda.device_count());\'; '

    cmd += f'python -u run_hydra.py {" ".join(params)}'

    sub += ['--wrap', f'\"{cmd}\"']
    print('Running command:', " ".join(sub))
    process = sps.Popen(" ".join(sub), shell=True, stdout=sps.PIPE)
    out, err = process.communicate()
    print(out.decode())
