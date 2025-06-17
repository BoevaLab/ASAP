import torch
from torch import nn
import numpy as np
import pathlib
from typing import List
import pyBigWig
from tqdm import tqdm
import json

from .dataloader import BaseDataset
from .trainer import Trainer, make_dataloader
from .utils.logger import TextLogger
from .models import VanillaCNN, CNN_LSTM, DilatedCNN, ConvNextTransformer, ConvNeXtCNN, ConvNeXtLSTM, ConvNeXtDCNN
from .snv import make_pcawg_df, add_predictions

def _get_model(model_name: str, use_map: bool = False):
    if model_name == 'cnn':
        return VanillaCNN(use_map=use_map)
    elif model_name == 'lstm':
        return CNN_LSTM(use_map=use_map)
    elif model_name == 'dcnn':
        return DilatedCNN(use_map=use_map)
    elif model_name == 'convnext_transformer':
        return ConvNextTransformer(use_map=use_map)
    elif model_name == 'convnext_cnn':
        return ConvNeXtCNN(use_map=use_map)
    elif model_name == 'convnext_lstm':
        return ConvNeXtLSTM(use_map=use_map)
    elif model_name == 'convnext_dcnn':
        return ConvNeXtDCNN(use_map=use_map)
    else:
        raise ValueError(f'Unknown model name: {model_name}')


def train_model(experiment_name : str, model: str, train_dataset: BaseDataset, val_dataset: BaseDataset, logs_dir: str, n_gpus: int=0, max_epochs: int=70, learning_rate: float=1e-3, batch_size: int=64, use_map: bool=False):
    """
    Train the model with the given datasets and parameters.
    
    Args:
        experiment_name (str): The name of the experiment.
        model (str): The model to train.
        train_dataset: The training dataset.
        val_dataset: The validation dataset.
        logs_dir (str): The directory to save logs.
        n_gpus (int): The number of GPUs to use for training.
        max_epochs (int): The maximum number of epochs to train.
        learning_rate (float): The learning rate for the optimizer.
        batch_size (int): The batch size for training.
        use_map (bool): Whether to use mappability for training.
    """
    # Check if gpu is available
    if n_gpus > 0 and not torch.cuda.is_available():
        n_gpus = 0
        print("No GPU available, using CPU instead.")
    
    # Count the number of GPUs available
    if n_gpus > torch.cuda.device_count():
        n_gpus = torch.cuda.device_count()
        print(f"Requested {n_gpus} GPUs, but only {torch.cuda.device_count()} are available. Using {n_gpus} GPUs instead.")

    # Initialize the model
    model = _get_model(model, use_map=use_map)

    # Initialize the trainer with the model and datasets
    trainer = Trainer(
        filename=experiment_name, 
        model=model,
        criterion=nn.PoissonNLLLoss(log_input=False),
        unmap_criterion=use_map,
        batch_size=batch_size,
        logger=TextLogger(logs_dir=logs_dir), 
        n_gpus=n_gpus,
    )

    # Start training
    trainer.fit(train_dset=train_dataset, val_dset=val_dataset, nr_epochs=max_epochs, learning_rate=learning_rate)

def eval_model(experiment_name: str, model: str, eval_dataset: BaseDataset, logs_dir: str, batch_size: int=64, use_map: bool=False):
    '''
    Evaluate the model on the given dataset.
    Args:
        experiment_name (str): The name of the experiment.
        model (str): The model to evaluate.
        eval_dataset: The evaluation dataset.
        logs_dir (str): The directory to load model checkpoints from.
        batch_size (int): The batch size for evaluation.
        use_map (bool): If mappability information was used during training.
    '''
    n_gpus = 1 if torch.cuda.is_available() else 0

    # Initialize the model
    model = _get_model(model, use_map=use_map)

    trainer = Trainer(
        filename=experiment_name, 
        model=model,
        criterion=nn.PoissonNLLLoss(log_input=False),
        unmap_criterion=use_map,
        batch_size=batch_size,
        logger=TextLogger(logs_dir=logs_dir), 
        n_gpus=n_gpus,
    )

    # for evaluation use the checkpoint of the best model
    print(f'Loading best model weights from {trainer.filename}')
    checkpoint_path = pathlib.Path(trainer.logger.logs_dir) / trainer.filename / 'checkpoint.pth'
    trainer.load_weights(checkpoint_path)
    
    test_chroms = eval_dataset.chroms
    scores = {}
    for chrom in test_chroms:
        eval_dataset.set_chroms([chrom])
        test_gen = make_dataloader(
            ddp_enabled=False,
            dataset=eval_dataset,
            batch_size=batch_size, 
            is_train=False
        )

        _, _, result_metrics = trainer.predict_and_evaluate(test_gen)
        scores[chrom] = {key: result_metrics[key] for key in ['pearson_r', 'mse', 'poisson_nll', 'spearman_r', 'kendall_tau']}
    return scores

def eval_robustness(experiment_name: str, model: str, eval_dataset: BaseDataset, logs_dir: str, batch_size: int=64, use_map: bool=False, nr_samples_for_var: int=17):
    '''
    Evaluate the robustness of the model on the given dataset.
    Args:
        experiment_name (str): The name of the experiment.
        model (str): The model to evaluate.
        eval_dataset: The evaluation dataset.
        logs_dir (str): The directory to save logs.
        batch_size (int): The batch size for evaluation.
        use_map (bool): Whether to use mappability for evaluation.
        nr_samples_for_var (int): The number of samples for variance calculation.
    '''
    # Fixed margin size for robustness evaluation
    margin = 768

    # Check if gpu is available
    n_gpus = 1 if torch.cuda.is_available() else 0
    
    # Initialize the model
    model = _get_model(model, use_map=use_map)

    trainer = Trainer(
        filename=experiment_name, 
        model=model,
        criterion=nn.PoissonNLLLoss(log_input=False),
        unmap_criterion=use_map,
        batch_size=batch_size,
        logger=TextLogger(logs_dir=logs_dir), 
        n_gpus=n_gpus,
    )

    # for evaluation use the checkpoint of the best model
    print(f'Loading best model weights from {trainer.filename}')
    checkpoint_path = pathlib.Path(trainer.logger.logs_dir) / trainer.filename / 'checkpoint.pth'
    trainer.load_weights(checkpoint_path)
    
    test_chroms = eval_dataset.chroms
    scores = {}
    for chrom in test_chroms:
        eval_dataset.set_chroms([chrom])
        test_gen = make_dataloader(
            ddp_enabled=False,
            dataset=eval_dataset,
            batch_size=batch_size // (nr_samples_for_var - 1), 
            is_train=False
        )

        _, _, cov, cov_per_bin = trainer.predict_robust_batch(test_gen, nr_samples_for_var=nr_samples_for_var, window=eval_dataset.window_size, margin=margin)
        cov_per_bin = np.nanmean(cov_per_bin, axis=0)
        scores[chrom] = {'cov': float(np.nanmean(cov)), 'cov_per_bin': {f'bin_{i}': float(cov_per_bin[i]) for i in range(len(cov_per_bin))}}
    return scores

def predict_snv_atac(experiment_name: str, model: str, snv_file: str, signal_file: str, logs_dir: str, out_dir: str, genome: str, chroms: List[int]=[*range(1,23)], use_map: bool=False, export_bigwig: str=None, scale: dict | float=1.0):
    """
    Predict ATAC-seq for SNVs using the trained model.
    Args:
        experiment_name (str): The name of the experiment.
        model (str): The model to evaluate.
        snv_file (str): The path to the SNV file.
        signal_file (str): The path to the signal file.
        logs_dir (str): The directory to save logs.
        out_path (str): The output path for predictions.
        genome (str): The reference genome.
        chroms (List[int]): List of chromosomes to predict on.
        use_map (bool): Whether to use mappability for model.
        export_bigwig (str): Export predictions as bigwig for "ref", "alt", or "both".
        scale (dict | float): Scaling factor for the predictions. If a dict, it should contain scaling factor corresponding to each chromosome.
    """
    window_size = 1024
    margin_size = 512
    bin_size = 4


    df = make_pcawg_df(snv_file)
    
    # Get file name from the snv file
    snv_file_name = pathlib.Path(snv_file).stem.split('.')[0]

    # Initialize the model
    model = _get_model(model, use_map=use_map)
    checkpoint_path = pathlib.Path(logs_dir) / experiment_name / 'checkpoint.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state_dict = torch.load(checkpoint_path, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # For loading a DDP model in non-DDP setting
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "module.")
        model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    # Add predictions to the dataframe
    results = add_predictions(
        snv=df,
        chroms=chroms,
        bw_file=signal_file,
        model=model,
        genome=genome,
        margin_size=margin_size,
        window_size=window_size,
        bin_size=bin_size,
        device=device,
    )

    # Save the results to a CSV file
    if not pathlib.Path(out_dir).exists():
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_file = pathlib.Path(out_dir) / f'{snv_file_name}_snv_predictions.csv'
    results.to_csv(out_file, index=False)

    if export_bigwig is not None:
        assert export_bigwig in ['ref', 'alt', 'both'], "export_bigwig must be one of ['ref', 'alt', 'both']"
        
        # Generate bigwig files
        print("Generating bigwig file...")
        chroms = results['chr'].unique()
        bw_in = pyBigWig.open(signal_file, 'r')

        # Check if scale is a dict or a float
        if isinstance(scale, dict):
            assert all(chrom in scale for chrom in chroms), "Scale dictionary must contain scaling factors for all chromosomes."
        elif isinstance(scale, (int, float)):
            scale = {chrom: scale for chrom in chroms}
        else:
            raise ValueError("Scale must be a dict or a float.")

        # Create output bigwig files
        ref_flg, alt_flg = False, False
        if export_bigwig == 'ref' or export_bigwig == 'both':
            bw_out_ref = pyBigWig.open(str(pathlib.Path(out_dir) / f'{snv_file_name}_ref.bw'), 'w')
            bw_out_ref.addHeader([(chrom_name, size) for chrom_name, size in bw_in.chroms().items()])
            ref_flg = True
        if export_bigwig == 'alt' or export_bigwig == 'both':
            bw_out_alt = pyBigWig.open(str(pathlib.Path(out_dir) / f'{snv_file_name}_alt.bw'), 'w')
            bw_out_alt.addHeader([(chrom_name, size) for chrom_name, size in bw_in.chroms().items()])
            alt_flg = True
        
        for chrom in tqdm(chroms):
            scale_factor = scale[chrom]
            if isinstance(scale_factor, (int, float)):
                scale_factor = float(scale_factor)
            else:
                raise ValueError(f"Scale factor for chromosome {chrom} must be a number.")
            

            chrom_df = results[results['chr'] == chrom]
            chrom_df = chrom_df.sort_values(by=['pos'])
            chrom_df = chrom_df.reset_index(drop=True)
            intervals = bw_in.intervals(chrom)
            starts = []
            ends = []
            if ref_flg:
                values1 = []
            if alt_flg:
                values2 = []

            interval_idx = 0
            region_idx = 0
            prev_region_end = -1

            while interval_idx < len(intervals) or region_idx < len(chrom_df):
                # Handle variant regions
                if region_idx < len(chrom_df):
                    region_start, region_end = chrom_df.iloc[region_idx]['pos']-window_size//2, chrom_df.iloc[region_idx]['pos']+window_size//2
                    # Add unmodified intervals before the region
                    while interval_idx < len(intervals) and intervals[interval_idx][0] < region_start:
                        i_start, i_end, i_value = intervals[interval_idx]
                        if i_start < prev_region_end:
                            i_start = prev_region_end
                        if i_end > region_start:
                            # Truncate interval if it overlaps the region
                            if i_start < region_start:
                                starts.append(i_start)
                                ends.append(region_start)
                                if ref_flg:
                                    values1.append(scale_factor*i_value)
                                if alt_flg:
                                    values2.append(scale_factor*i_value)
                        else:
                            if i_start < region_start:
                                starts.append(i_start)
                                ends.append(i_end)
                                if ref_flg:
                                    values1.append(scale_factor*i_value)
                                if alt_flg:
                                    values2.append(scale_factor*i_value)
                        interval_idx += 1

                    # Add the modified region (x-512 to x+512) with vector y
                    if ref_flg:
                        y1 = [x for y in json.loads(chrom_df.iloc[region_idx]['signal_pred_ref']) for x in y]
                        y1 = np.exp(np.array(y1))-1
                        y1 = y1[margin_size//bin_size:-margin_size//bin_size]
                    if alt_flg:
                        y2 = [x for y in json.loads(chrom_df.iloc[region_idx]['signal_pred_alt']) for x in y]
                        y2 = np.exp(np.array(y2))-1
                        y2 = y2[margin_size//bin_size:-margin_size//bin_size]
                    for i in range(margin_size//2):
                        bin_start = region_start + i * bin_size
                        bin_end = bin_start + bin_size
                        starts.append(bin_start)
                        ends.append(bin_end)
                        if ref_flg:
                            values1.append(scale_factor*y1[i])
                        if alt_flg:
                            values2.append(scale_factor*y2[i])
                    region_idx += 1

                    # Skip input intervals that overlap the modified region
                    while interval_idx < len(intervals) and intervals[interval_idx][1] <= region_end:
                        interval_idx += 1

                    prev_region_end = region_end

                # Copy remaining intervals
                else:
                    while interval_idx < len(intervals):
                        i_start, i_end, i_value = intervals[interval_idx]
                        if i_start < prev_region_end:
                            i_start = prev_region_end
                        starts.append(i_start)
                        ends.append(i_end)
                        if ref_flg:
                            values1.append(scale_factor*i_value)
                        if alt_flg:
                            values2.append(scale_factor*i_value)
                        interval_idx += 1

            # Write intervals for this chromosome
            if starts:
                # Sort by start position to ensure valid bigWig format
                sorted_indices = np.argsort(starts)
                starts = [starts[i] for i in sorted_indices]
                ends = [ends[i] for i in sorted_indices]
                if ref_flg:
                    values1 = [values1[i] for i in sorted_indices]
                    bw_out_ref.addEntries([chrom] * len(starts), starts, ends=ends, values=values1)    
                if alt_flg:
                    values2 = [values2[i] for i in sorted_indices]
                    bw_out_alt.addEntries([chrom] * len(starts), starts, ends=ends, values=values2)
                
        # Close files
        bw_in.close()
        if ref_flg:
            bw_out_ref.close()
        if alt_flg:
            bw_out_alt.close()
        


def export_predictions(experiment_name: str, model: str, eval_dataset: BaseDataset, logs_dir: str, out_dir: str, batch_size: int=64, use_map: bool=False):
    """
    Export the predictions to a file.
    Args:
        experiment_name (str): The name of the experiment.
        model (str): The model to evaluate.
        eval_dataset: The evaluation dataset.
        logs_dir (str): The directory to save logs.
        out_dir (str): The output directory for predictions.
        batch_size (int): The batch size for evaluation.
        use_map (bool): Whether to use mappability for evaluation.
    """
    n_gpus = 1 if torch.cuda.is_available() else 0

    # Initialize the model
    model = _get_model(model, use_map=use_map)

    trainer = Trainer(
        filename=experiment_name, 
        model=model,
        criterion=nn.PoissonNLLLoss(log_input=False),
        unmap_criterion=use_map,
        batch_size=batch_size,
        logger=TextLogger(logs_dir=logs_dir), 
        n_gpus=n_gpus,
    )

    # for evaluation use the checkpoint of the best model
    print(f'Loading best model weights from {trainer.filename}')
    checkpoint_path = pathlib.Path(trainer.logger.logs_dir) / trainer.filename / 'checkpoint.pth'
    trainer.load_weights(checkpoint_path)
    
    test_chroms = eval_dataset.chroms
    
    for chrom in test_chroms:
        eval_dataset.set_chroms([chrom], reset_unmap=True)
        file_name = f'{experiment_name}_chr{chrom}.bw'
        file_name = pathlib.Path(out_dir) / file_name
        print('Saving bigwig file:', file_name)
        test_gen = make_dataloader(
            ddp_enabled=False,
            dataset=eval_dataset,
            batch_size=batch_size, 
            is_train=False
        )

        _, predictions, _ = trainer.predict_and_evaluate(test_gen)
        predictions = np.exp(predictions)-1
        predictions = np.reshape(predictions, (-1,256))   # Total 256 bins predicted per input
        predictions = np.expand_dims(predictions, axis=0)   # Converting the array from shape (...) to (1, ...) to account for single chromosome
        eval_dataset.write_predictions_to_bigwig(str(file_name), predictions)

