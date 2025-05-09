import random
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from asap.dataloader import bw_to_data
from asap.dataloader.augmentation import ReverseComplement
import pathlib

class BaseDataset(Dataset):
    def __init__(
        self,
        genome: str,
        signal_files: str,
        chroms: List[int],
        window_size: int,
        margin_size: int,
        bin_size: int = 100,
        random_shift=False,
        augmentations=False,
        lower_bound=None,
        blacklist_file=None,
        unmap_file=None,
        unmap_threshold=None,
        logspace=False,
        output_format="ohe",
        memmap=False,
        generated=None,
        is_train=False,
        is_robustness=False,
    ):
        super().__init__()
        self.genome = genome
        self.signal_files = [signal_files]

        # if we randomly shift to augment the data
        self.random_shift = random_shift
        
        if is_train:
            self.window_size = window_size + 2*margin_size
            self.margin_size = 0
        elif is_robustness:
            # 1)              |---m---|-----w-----|---m---| # input size
            # 2) |-----p-----|---------vc---------|-----p-----| # valid center + padding, vc=input_size//2
            #             |-----p-----|---------vc---------|-----p-----|
            # 3) |--------------------------t'-------------------------| # total size
            # 4) |---------m'---------|-----w-----|---------m'---------| # margin_size' 
            input_size = window_size + 2 * margin_size # 1)
            valid_center = input_size // 2 # 2), per definition
            total_size = input_size + valid_center - window_size # 2  * (padding + valid_center) - window_size = total_size
            self.window_size = window_size
            # preprocessing margin size is determined by the total size
            self.margin_size = (total_size - window_size) // 2
        else:
            self.window_size = window_size
            self.margin_size = margin_size

        # if yes --> need bigger windows for slicing
        self.pre_process_window_size = (
            self.window_size if not random_shift else self.window_size * 2
        )

        self.bin_size = bin_size
        self.chroms = chroms
        self.lower_bound = lower_bound
        self.unmap_file = unmap_file
        self.blacklist_file = blacklist_file
        self.unmap_threshold = unmap_threshold
        self.logspace = logspace
        if output_format not in ["ohe", "idx", "str"]:
            raise ValueError

        self.output_format = output_format
        self.memmap = memmap
        
        augmentations = [ReverseComplement] if augmentations else []
        self.augmentations = [aug() for aug in augmentations]

        self.generated = generated

    def setup(self):
        pathlib.Path(self.generated).mkdir(parents=True, exist_ok=True)
        # chroms have been grouped by amount of bounded data per chromosome to balance folds
        self.chrom_lengths = []
        self.X, self.y, self.seq_starts = [], [], []
        for chrom in self.chroms:
            X_, y_, seq_starts_ = self._generate_chrom_data(chrom)
            self.chrom_lengths.append(len(X_))
            self.X.append(X_)
            self.y.append(y_)
            self.seq_starts.append(seq_starts_)
        self.cum_chrom_lengths = np.cumsum(self.chrom_lengths)

    def set_chroms(self, chroms, reset_unmap=False):
        self.chroms = chroms
        if reset_unmap:
            self.unmap_file = None
        self.setup()

    def __len__(self):
        return self.cum_chrom_lengths[-1]

    def idx_to_ohe(self, idx: np.ndarray) -> np.ndarray:
        eye = np.concatenate(
            (np.eye(4, dtype=idx.dtype), np.zeros((1, 4), dtype=idx.dtype)), axis=0
        )
        one_hot_encoded = eye[idx]
        return one_hot_encoded

    def idx_to_seq(self, idx: np.ndarray) -> np.ndarray:
        seq = np.array(["A", "G", "C", "T", "N"])[idx]
        return "".join(seq)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        # get index of first chromosome where cumsum is bigger than index
        # This is the chromosome that we want to pull from
        chrom_idx = np.argmax([self.cum_chrom_lengths > index])
        # offset index by cumsum to get index within chromosome
        if chrom_idx != 0:
            idx = index - self.cum_chrom_lengths[chrom_idx - 1]
        else:
            idx = index
        if self.random_shift:
            shift = random.randint(
                0, self.window_size // self.bin_size
            ) # take a random window within step_size
            x_shift = shift * self.bin_size
            X = self.X[chrom_idx][
                idx, x_shift : x_shift + (self.window_size + 2 * self.margin_size)
            ]
            y = self.y[chrom_idx][
                idx, shift : shift + (self.window_size // self.bin_size)
            ]
        else:
            X = self.X[chrom_idx][idx]
            y = self.y[chrom_idx][idx]

        if self.logspace:
            y = np.log(y + 1)

        m = X[..., [1]].astype(y.dtype)
        X = X[..., 0]

        for aug in self.augmentations:
            X, y = aug(X, y)

        if self.output_format == "ohe":
            X = self.idx_to_ohe(X).astype(np.float32)
        elif self.output_format == "str":
            X = self.idx_to_seq(X)
        
        # torch doesn't like non-writeable tensors
        X = torch.from_numpy(np.copy(X))
        m = torch.from_numpy(np.copy(m))
        y = torch.from_numpy(np.copy(y))
        return X, m, y


    def _generate_chrom_data(self, chrom: int) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _properties_str(self) -> str:
        return f"{self.bin_size}_{self.pre_process_window_size}_{self.step_size}_{len(self.signal_files)}"
    
    def write_predictions_to_bigwig(self, file_name: str, preds: np.ndarray) -> None:
        '''
        lengths = [0] + list(self.cum_chrom_lengths)
        print(np.array(self.seq_starts).shape)
        preds = [
            preds[lengths[i] : lengths[i + 1], ...]
            for i, _ in enumerate(self.chroms)
        ]
        if not all([preds[i].shape == y[..., 0].shape for i, y in enumerate(self.y)]):
            print(preds[0].shape, self.y[0][...,0].shape)
            raise ValueError("Predictions must have same shape as y")
        '''
        bw_to_data.write_predictions_to_bigwig(
            file_name, preds, self.chroms, self.seq_starts, self.genome, self.bin_size
        )
