import numpy as np
import pandas as pd
import pyBigWig
import errno
import os
from pathlib import Path
from typing import Tuple, Dict


def assure_folders(folders):
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
    print(f'Using folders at {[os.path.abspath(f) for f in folders]}')


def get_bw_from_file(signal_file: str, file_mode='r'):
    if 'r' in file_mode and not os.path.isfile(signal_file):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), signal_file)
    if file_mode in ['r', 'w']:
        bw = pyBigWig.open(signal_file, file_mode)
    else:
        raise ValueError(f"file_mode {file_mode} is not valid.")
    return bw


def get_peak_locations(bed_file: str, chrom: int) -> np.ndarray:
    df = pd.read_csv(bed_file, delimiter='\t', header=None,
                     names=['chrom', 'chrom_start', 'chrom_end', 'offset'], usecols=[0, 1, 2, 9])
    df = df[df['chrom'] == f'chr{chrom}']
    return df['offset'].to_numpy() + df['chrom_start'].to_numpy()


def npz_to_np(file_name: str) -> Tuple[np.ndarray, np.ndarray]:
    print('Loading: ', file_name)
    npz_file = np.load(file_name, allow_pickle=True)
    x, y = npz_file['x'].astype('float16'), npz_file['y'].astype('float16')
    print(f'Loaded data: x={x.shape}, y={y.shape} from {file_name}')
    return x, y


def np_to_npz(file_name: str, x: np.ndarray, y: np.ndarray) -> None:
    print(f'Saving to {file_name}')
    if y.shape[-1] == 1:
        # squeeze in case we have single task prediction
        y = np.squeeze(y)
    np.savez(file_name, x=x, y=y)


def add_to_chrom_npz(file_name: str, data_to_save: Dict[str, np.ndarray]) -> None:
    print(f'Saving predictions for chroms {list(data_to_save.keys())} to {file_name}')
    data = {}
    if os.path.exists(file_name):
        print(file_name)
        data = dict(np.load(file_name, allow_pickle=True))
    data.update(data_to_save)
    np.savez_compressed(file_name, **data)


def get_chrom_npz(file_name: str) -> dict:
    return dict(np.load(file_name, allow_pickle=True))


def append_results_to_csv(file_path: str, row_data: dict):
    df = pd.DataFrame([row_data])
    df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)
