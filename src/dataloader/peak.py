import numpy as np
from src.dataloader import bw_to_data
from src.dataloader.base import BaseDataset
from src.dataloader.utils.io import get_peak_locations
from src.config import CellLineConfig


class PeakDataset(BaseDataset):
    def __init__(
        self,
        genome: str,
        signal_files: str,
        bed_files: str,
        chroms: list[int],
        window_size: int,
        margin_size: int,
        bin_size: int = 100,
        random_shift=False,
        augmentations=False,
        lower_bound=None,
        unmap_file=None,
        blacklist_file=None,
        unmap_threshold=None,
        logspace=False,
        output_format="ohe",
        memmap=False,
        generated=None,
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
            unmap_file=unmap_file,
            blacklist_file=blacklist_file,
            unmap_threshold=unmap_threshold,
            logspace=logspace,
            output_format=output_format,
            memmap=memmap,
            generated=generated,
            is_robustness=is_robustness,
        )
        self.bed_files = [bed_files]
        assert self.unmap_threshold == 0
        if self.chroms:    
            self.setup()

    def _generate_chrom_data(self, chrom: int) -> tuple[np.ndarray, np.ndarray]:
        peak_centers = _get_merged_peaks(self.bed_files, chrom)
        seq_starts = peak_centers - self.pre_process_window_size // 2
        X, y, seq_starts = bw_to_data.idx_to_filtered_data(
            genome=self.genome,
            signal_files=self.signal_files,
            chrom=chrom,
            seq_starts=seq_starts,
            window_size=self.pre_process_window_size,
            margin_size=self.margin_size,
            bin_size=self.bin_size,
            lower_bound=self.lower_bound,
            blacklist_bed_files=[self.blacklist_file],
            unmappable_bed_file=self.unmap_file,
            unmap_threshold=self.unmap_threshold,
            memmap=self.memmap,
            generated=self.generated,
        )
        return X, y, seq_starts


def _get_merged_peaks(
    bed_files: list[str], chrom: int, merge_peaks_within: int = 200
) -> np.ndarray:
    all_peaks = np.empty(0)
    for bed_file in bed_files:
        all_peaks = np.append(all_peaks, get_peak_locations(bed_file, chrom))
    all_peaks.sort()
    print(f"Merging. All peaks: {all_peaks.shape}")

    diffs = all_peaks[1:] - all_peaks[:-1]
    new_peaks = []

    current = 0
    merge_from = 0
    for diff, peak in zip(diffs, all_peaks[:-1]):
        current += 1
        if diff > merge_peaks_within:
            new_peaks.append(int(np.mean(all_peaks[merge_from:current])))
            merge_from = current
    print(
        f"Reduced all peaks ({len(all_peaks)}) into new peaks ({len(new_peaks)}) "
        f"by merging peaks within {merge_peaks_within}"
    )
    return np.array(new_peaks)
