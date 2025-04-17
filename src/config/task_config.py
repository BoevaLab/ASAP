import json

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
from src.config import CONFIG, CellLineConfig, ModelConfig


class TaskConfig:
    def __init__(self, cfg: DictConfig):
        print('Running with config:')
        print(OmegaConf.to_yaml(cfg))
        self.cfg = cfg

        bench_config = dict(cfg.bench_settings)
        bench_config.update(instantiate(cfg.paths))
        CONFIG.setup(**bench_config)

        necessary_fields = ['cell_line', 'experiment_name', 'model']
        for field in necessary_fields:
            if cfg[field] is None:
                print(f'CONFIG MISSING! Field: {field}')

        self.cell_line: CellLineConfig = instantiate(cfg.cell_line)
        if cfg['model'] is not None:
            self.model: ModelConfig = instantiate(cfg.model)

        if torch.cuda.is_available():
            self.nr_devices = torch.cuda.device_count()
            print(f'Running with {self.nr_devices} GPUs')
        else:
            self.nr_devices = 0
            print('Running on CPU')

        assert cfg.run_script in cfg, f'Task configuration missing for run_script={cfg.run_script}'
        
        self.task_params = instantiate(cfg[cfg.run_script])
        print('config\n', self.task_params)

        self.eval_datasets = instantiate(cfg['eval_datasets'])
        self.robustness_datasets = instantiate(cfg['robustness_datasets'])
