from src.config import TaskConfig, CONFIG
from src.trainer import Trainer
import torch.multiprocessing as mp
from src.utils.logger import get_default_logger
import torch.nn as nn
from omegaconf import OmegaConf


def assert_inputs(task_config: TaskConfig) -> None:
    assert task_config.cfg.experiment_name is not None, 'Undefined training experiment name'


def run(task_config: TaskConfig) -> None:
    assert_inputs(task_config)

    model_cfg = task_config.model
    params = task_config.task_params
    use_map = model_cfg.use_map
    
    wandb_cfg = {
        'project': 'mutiger-snv-paper',
        'entity': 'sdsc-paired-hydro',
        'config': {'cfg': OmegaConf.to_container(task_config.cfg, resolve=True)}
    }

    trainer = Trainer(
        filename=task_config.cfg.experiment_name,
        model=model_cfg.model,
        checkpoint_name=params.save_file,
        criterion=nn.PoissonNLLLoss(log_input=False),
        batch_size=model_cfg.batch_size,
        unmap_criterion=use_map,
        logger=get_default_logger(CONFIG.logs, wandb_cfg=wandb_cfg),
        task_config=task_config
    )

    train_chroms, val_chroms, test_chroms = CONFIG.all_chroms, [7,8,9], []

    train_dataset = params.train_dataset
    val_dataset = params.val_dataset
    train_dataset.set_chroms(train_chroms)
    val_dataset.set_chroms(val_chroms)

    trainer.fit(
        train_dset=train_dataset,
        val_dset=val_dataset,
        nr_epochs=params.trainer.max_epochs,
        learning_rate=model_cfg.learning_rate
    )
