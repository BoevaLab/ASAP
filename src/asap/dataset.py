from .dataloader import WGDataset, PeakDataset
from typing import List

def training_datasets(signal_file: str, genome: str, train_chroms: List[int], val_chroms: List[int], generated: str, blacklist_file: List[str] = None, unmap_file: str = None):
    '''
    Create training and validation datasets for the model.
    Args:
        signal_file (str): Path to the signal file.
        genome (str): Path to the genome file.
        train_chroms (List[int]): List of chromosomes for training.
        val_chroms (List[int]): List of chromosomes for validation.
        generated (str): Path to the generated data.
        blacklist_file (List[str]): List of paths to blacklist files (including SNVs).
        unmap_file (str): Path to the unmapped regions file.
    '''
    
    train_dataset = WGDataset(
        genome=genome,
        signal_files=signal_file,
        chroms=None,
        window_size=1024,
        margin_size=512,
        step_size=1024,
        bin_size=4,
        random_shift=True,
        augmentations=True,
        blacklist_file=blacklist_file,
        unmap_file=unmap_file,
        unmap_threshold=0.35,
        logspace=True,
        output_format="ohe",
        memmap=False,
        generated=generated,
        is_train=True,
        is_robustness=False,
    )
    train_dataset.set_chroms(train_chroms)

    val_dataset = WGDataset(
        genome=genome,
        signal_files=signal_file,
        chroms=None,
        window_size=1024,
        margin_size=512,
        step_size=None,
        bin_size=4,
        random_shift=False,
        augmentations=False,
        blacklist_file=blacklist_file,
        unmap_file=unmap_file,
        unmap_threshold=0,
        logspace=True,
        output_format="ohe",
        memmap=False,
        generated=generated,
        is_train=False,
        is_robustness=False,
    )
    val_dataset.set_chroms(val_chroms)

    return train_dataset, val_dataset

def peak_dataset(signal_file: str, peak_file: str, genome: str, chroms: List[int], generated: str, blacklist_file: List[str] = None, unmap_file: str = None):
    '''
    Create a peak dataset for evaluation.
    Args:
        signal_file (str): Path to the signal file.
        peak_file (str): Path to the peak file.
        genome (str): Path to the genome file.
        chroms (List[int]): List of chromosomes for evaluation.
        generated (str): Path to the generated data.
        blacklist_file (List[str]): List of paths to blacklist files (including SNVs).
        unmap_file (str): Path to the unmapped regions file.
    '''
    dataset = PeakDataset(
        genome=genome,
        signal_files=signal_file,
        bed_files=peak_file,
        chroms=None,
        window_size=1024,
        margin_size=512,
        bin_size=4,
        random_shift=False,
        augmentations=False,
        lower_bound=None,
        unmap_file=unmap_file,
        blacklist_file=blacklist_file,
        unmap_threshold=0,
        logspace=True,
        output_format="ohe",
        memmap=False,
        generated=generated,
        is_robustness=False,
    )
    dataset.set_chroms(chroms)

    return dataset

def robustness_peak_dataset(signal_file: str, peak_file: str, genome: str, chroms: List[int], generated: str, blacklist_file: List[str] = None, unmap_file: str = None):
    '''
    Create a peak dataset for robustness evaluation.
    Args:
        signal_file (str): Path to the signal file.
        peak_file (str): Path to the peak file.
        genome (str): Path to the genome file.
        chroms (List[int]): List of chromosomes for evaluation.
        generated (str): Path to the generated data.
        blacklist_file (List[str]): List of paths to blacklist files (including SNVs).
        unmap_file (str): Path to the unmapped regions file.
    '''
    dataset = PeakDataset(
        genome=genome,
        signal_files=signal_file,
        bed_files=peak_file,
        chroms=None,
        window_size=512,
        margin_size=768,
        bin_size=4,
        random_shift=False,
        augmentations=False,
        lower_bound=None,
        unmap_file=unmap_file,
        blacklist_file=blacklist_file,
        unmap_threshold=0,
        logspace=True,
        output_format="ohe",
        memmap=False,
        generated=generated,
        is_robustness=True,
    )
    dataset.set_chroms(chroms)

    return dataset

def wg_dataset(signal_file: str, genome: str, chroms: List[int], generated: str, blacklist_file: List[str] = None, unmap_file: str = None):
    '''
    Create a whole genome dataset for evaluation.
    Args:
        signal_file (str): Path to the signal file.
        genome (str): Path to the genome file.
        chroms (List[int]): List of chromosomes for evaluation.
        generated (str): Path to the generated data.
        blacklist_file (List[str]): List of paths to blacklist files (including SNVs).
        unmap_file (str): Path to the unmapped regions file.
    '''
    dataset = WGDataset(
        genome=genome,
        signal_files=signal_file,
        chroms=None,
        window_size=1024,
        margin_size=512,
        step_size=None,
        bin_size=4,
        random_shift=False,
        augmentations=False,
        blacklist_file=blacklist_file,
        unmap_file=unmap_file,
        unmap_threshold=0,
        logspace=True,
        output_format="ohe",
        memmap=False,
        generated=generated,
        is_train=False,
        is_robustness=False,
    )
    dataset.set_chroms(chroms)

    return dataset

def robustness_wg_dataset(signal_file: str, genome: str, chroms: List[int], generated: str, blacklist_file: List[str] = None, unmap_file: str = None):
    '''
    Create a whole genome dataset for robustness evaluation.
    Args:
        signal_file (str): Path to the signal file.
        genome (str): Path to the genome file.
        chroms (List[int]): List of chromosomes for evaluation.
        generated (str): Path to the generated data.
        blacklist_file (List[str]): List of paths to blacklist files (including SNVs).
        unmap_file (str): Path to the unmapped regions file.
    '''
    dataset = WGDataset(
        genome=genome,
        signal_files=signal_file,
        chroms=None,
        window_size=512,
        margin_size=768,
        step_size=None,
        bin_size=4,
        random_shift=False,
        augmentations=False,
        blacklist_file=blacklist_file,
        unmap_file=unmap_file,
        unmap_threshold=0,
        logspace=True,
        output_format="ohe",
        memmap=False,
        generated=generated,
        is_train=False,
        is_robustness=True,
    )
    dataset.set_chroms(chroms)

    return dataset