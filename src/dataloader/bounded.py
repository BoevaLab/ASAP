import numpy as np

from src.dataloader import bw_to_data
from src.dataloader.base import BaseDataset


class BoundedDataset(BaseDataset):
    def __init__(
        self,
        genome: str,
        signal_files: str,
        chroms: list[int],
        window_size: int,
        margin_size: int,
        step_size: int = None,
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
        super().__init__(
            genome=genome,
            signal_files=signal_files,
            chroms=chroms,
            window_size=window_size,
            margin_size=margin_size,
            bin_size=bin_size,
            random_shift=random_shift,
            augmentations=augmentations,
            lower_bound=lower_bound,
            blacklist_file=blacklist_file,
            unmap_file=unmap_file,
            unmap_threshold=unmap_threshold,
            logspace=logspace,
            output_format=output_format,
            memmap=memmap,
            generated=generated,
            is_train=is_train,
            is_robustness=is_robustness,
        )
        self.step_size = step_size if step_size is not None else self.pre_process_window_size
        if self.chroms:
            self.setup()

    def _generate_chrom_data(self, chrom: int) -> tuple[np.ndarray, np.ndarray]:
        X, y, seq_starts = bw_to_data.get_wg_filtered_data(
            genome=self.genome,
            signal_files=self.signal_files,
            chrom=chrom,
            window_size=self.pre_process_window_size,
            margin_size=self.margin_size,
            step_size=self.step_size,
            bin_size=self.bin_size,
            lower_bound=self.lower_bound,
            blacklist_bed_files=[self.blacklist_file],
            unmappable_bed_file=self.unmap_file,
            unmap_threshold=self.unmap_threshold,
            memmap=self.memmap,
            generated=self.generated,
        )
        return X, y, seq_starts