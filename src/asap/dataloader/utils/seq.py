import numpy as np
from pyfaidx import Fasta
from typing import Tuple


def get_chr_seq(genome: str, chrom: int) -> np.ndarray:
    print(f'Loading genome file: {genome}')
    genome_seq = Fasta(genome)
    chr_seq = genome_seq[f'chr{chrom}'][0:-1].seq
    return seq_to_idx(chr_seq)

def seq_to_idx(seq: str) -> np.array:
    seq_array = np.array(list(seq.upper()))
    mapping = {'A': 0, 'G': 1, 'C': 2, 'T': 3, 'N': 4}
    indices = np.vectorize(mapping.get)(seq_array).astype(np.int8)
    return indices

def seq_to_onehot(seq: str) -> np.array:
    seq_array = np.array(list(seq.upper()))
    mapping = {'A': 0, 'G': 1, 'C': 2, 'T': 3, 'N': 4}
    indices = np.vectorize(mapping.get)(seq_array)
    eye = np.concatenate((np.eye(4), np.zeros((1, 4))), axis=0)
    one_hot_encoded = eye[indices]
    return one_hot_encoded


# def seq_to_onehot(seq: str) -> np.array:
#     mapping = {'A': [1, 0, 0, 0], 'G': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
#     return np.array([np.array(mapping[char], dtype=np.float32) for char in seq.upper()])


def onehot_to_seq(onehot: np.ndarray) -> str:
    by_index = ['A', 'G', 'C', 'T']
    return ''.join(list(map(lambda el: by_index[np.arange(4)[el == 1][0]], onehot)))


def get_chr_range(chr_seq: str) -> Tuple[int, int]:
    chr_len = len(chr_seq)

    chrom_start = min([chr_seq.find(char) for char in 'ATCG'])
    rev_chr_seq = chr_seq[::-1]
    chrom_end = chr_len - min([rev_chr_seq.find(char) for char in 'ATCG'])
    return chrom_start, chrom_end


def get_range_by_chrom_number(genome, chrom: int, divisible_by: int = None) -> Tuple[int, int]:
    genome_seq = Fasta(genome)
    chr_seq = genome_seq[f'chr{chrom}'][0:-1].seq.upper()
    start, end = get_chr_range(chr_seq)
    if divisible_by is not None:
        end -= (end - start) % divisible_by
    return start, end
