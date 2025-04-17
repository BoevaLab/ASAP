import numpy as np
from omegaconf import OmegaConf
from datetime import datetime

from src.config import TaskConfig
from src.dataloader import BaseDataset
from src.dataloader.utils.io import add_to_chrom_npz, append_results_to_csv
from src.trainer import Trainer, make_dataloader
import pathlib

def generate_bigwig(task_config: TaskConfig, trainer: Trainer, test_chrom: int, experiment_data: dict):
    out_path = pathlib.Path(f'{task_config.cfg.paths.dir.generated}/predictions/')
    out_path.mkdir(parents=True, exist_ok=True)
    # make predictions for the evaluation datasets defined in config
    datasets: dict[str, BaseDataset] = OmegaConf.to_container(task_config.eval_datasets) 
    chrom = test_chrom
    eval_dataset = datasets['wg']
    eval_dataset.set_chroms([chrom], reset_unmap=True)
    file_name = out_path / f'{task_config.cell_line.exp_name}_{task_config.model.name}_chr{chrom}.bw'
    print('Saving bigwig:', file_name)
    valid_chrom_gen = make_dataloader(
        ddp_enabled=False,
        dataset=eval_dataset,
        batch_size=task_config.model.batch_size, 
        is_train=False
    )
    '''
    pred_file = out_path / f'eval_wg_{task_config.cell_line.exp_name}_{task_config.model.name}_{task_config.cfg.experiment_name}_pred.npz'
    preds = dict(np.load(pred_file, allow_pickle=True))
    predictions = preds[f'chr{test_chrom}']
    '''
    print('Start:', datetime.now())
    _, predictions, _ = trainer.predict_and_evaluate(valid_chrom_gen)
    print('End:', datetime.now())
    predictions = np.exp(predictions)-1
    predictions = np.reshape(predictions, (-1,256))   # Total 256 bins predicted per input
    predictions = np.expand_dims(predictions, axis=0)   # Converting the array from shape (...) to (1, ...) to account for single chromosome
    print('Predictions shape:', predictions.shape)
    eval_dataset.write_predictions_to_bigwig(str(file_name), predictions, chrom)



def eval_on_datasets(task_config: TaskConfig, trainer: Trainer, val_chroms: list[int], experiment_data: dict):
    out_path = pathlib.Path(f'{task_config.cfg.paths.dir.generated}/predictions/')
    out_path.mkdir(parents=True, exist_ok=True)
    # make predictions for the evaluation datasets defined in config
    datasets: dict[str, BaseDataset] = OmegaConf.to_container(task_config.eval_datasets) 
    for dataset_name, eval_dataset in datasets.items():
        for chrom in val_chroms:
            eval_dataset.set_chroms([chrom])
            file_base = out_path / f'eval_{dataset_name}_{task_config.cell_line.exp_name}'
            print('Adding predictions to file:', file_base)
            valid_chrom_gen = make_dataloader(
                ddp_enabled=False,
                dataset=eval_dataset,
                batch_size=task_config.model.batch_size, 
                is_train=False
            )

            true, predictions, result_metrics = trainer.predict_and_evaluate(valid_chrom_gen)

            result_data = dict(dataset=dataset_name, chrom=chrom)
            append_results_to_csv(file_path=task_config.cfg.paths.predictions.scores,
                                  row_data={**experiment_data, **result_data, **result_metrics})

            add_to_chrom_npz(file_name=f'{file_base}_{task_config.model.name}_{task_config.cfg.experiment_name}_true.npz', data_to_save={f'chr{chrom}': true})
            add_to_chrom_npz(file_name=f'{file_base}_{task_config.model.name}_{task_config.cfg.experiment_name}_pred.npz',
                             data_to_save={f'chr{chrom}': predictions})


def eval_robustness(task_config: TaskConfig, trainer: Trainer, val_chroms: list[int], experiment_data: dict, nr_samples_for_var=17):
    out_path = pathlib.Path(f'{task_config.cfg.paths.dir.generated}/robustness/')
    out_path.mkdir(parents=True, exist_ok=True)
    datasets: dict[str, BaseDataset] = OmegaConf.to_container(task_config.robustness_datasets)
    for dataset_name, rob_dataset in datasets.items():
        for chrom in val_chroms:
            rob_dataset.set_chroms([chrom])
            test_gen = make_dataloader(
                ddp_enabled=False,
                dataset=rob_dataset,
                batch_size=task_config.model.batch_size // (nr_samples_for_var - 1), 
                is_train=False
            )
            true, predictions, variance, variance_per_bin = trainer.predict_robust_batch(
                test_gen,
                nr_samples_for_var=nr_samples_for_var,
                window=task_config.cfg.robustness_datasets[dataset_name].window_size,
                margin=task_config.cfg.robustness_datasets[dataset_name].margin_size
            )
            print(variance.shape, f'NaN proportion: {np.isnan(variance).sum() / len(variance):.2f}')
            variance = float(np.nanmean(variance))
            variance_per_bin = np.nanmean(variance_per_bin, axis=0)
            variance_per_bin = {f'bin_{i}': float(variance_per_bin[i]) for i in range(len(variance_per_bin))}

            row_data = {**experiment_data, **dict(dataset=dataset_name, chrom=chrom, variance=variance), **variance_per_bin}
            append_results_to_csv(file_path=task_config.cfg.paths.predictions.robustness, row_data=row_data)

            file_base = out_path / f'eval_{dataset_name}_{task_config.cell_line.exp_name}'
            print('Adding predictions to file:', file_base)
            add_to_chrom_npz(file_name=f'{file_base}_{task_config.model.name}_{task_config.cfg.experiment_name}_true.npz', data_to_save={f'chr{chrom}': true})
            add_to_chrom_npz(file_name=f'{file_base}_{task_config.model.name}_{task_config.cfg.experiment_name}_pred.npz',
                             data_to_save={f'chr{chrom}': predictions})
