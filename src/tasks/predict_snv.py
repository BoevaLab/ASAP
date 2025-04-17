from src.config import TaskConfig, CONFIG
from src.models.load_model import get_checkpoints_all_folds, get_checkpoint_all_chroms
from src.snv.data_loader import get_pcawg_df
from src.snv.predict import add_predictions
import torch
from omegaconf import OmegaConf


def add_cv_predictions(task_config: TaskConfig, df):
    cfg = task_config.cfg
    model_cfg = task_config.model
    cell_line = task_config.cell_line

    # 2. load all CV checkpoints
    checkpoints = get_checkpoints_all_folds(
        checkpoint_dir=cfg.paths.dir.logs,
        model_name=model_cfg.name,
        train_exp=cfg.experiment_name,
        cell_line_exp=cell_line.exp_name
    )
    for fold, checkpoint in enumerate(checkpoints):
        if checkpoint is None:
            continue
        print(f'Adding predictions for {cell_line.exp_name} chroms in fold {fold} with model checkpoint: {checkpoint}')
        model_cfg.model.load_weights(checkpoint)

        # 3. add predictions for each fold
        add_predictions(df, cell_line=cell_line, model_cfg=model_cfg, chroms=CONFIG.folds[fold])


def load_weights(cfg, path, device):
    state_dict = torch.load(path, map_location=device)
    try:
        cfg.model.load_state_dict(state_dict)
    except RuntimeError:
        # For loading a DDP model in non-DDP setting
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "module.")
        cfg.model.load_state_dict(state_dict)
    cfg.model.eval()
    return cfg


def run(task_config: TaskConfig) -> None:
    cfg = task_config.cfg
    model_cfg = task_config.model
    cell_line = task_config.cell_line
    params = task_config.task_params
    genome = CONFIG.genome
    margin_size = OmegaConf.to_container(task_config.eval_datasets)['peaks'].margin_size
    window_size = OmegaConf.to_container(task_config.eval_datasets)['peaks'].window_size
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 1. load mutations
    df = get_pcawg_df(cell_line, output_path=params.output_path)

    # 2. if we have all chroms model use that
    checkpoint = get_checkpoint_all_chroms(
        checkpoint_dir=cfg.paths.dir.logs,
        model_name=model_cfg.name,
        train_exp=cfg.experiment_name,
        cell_line_exp=cell_line.exp_name
    )                                                   # checkpoint is the path to directory that contains "checkpoint.pth"
    if checkpoint:
        checkpoint_file = f'{checkpoint}/checkpoint.pth'
        model_cfg = load_weights(model_cfg, checkpoint_file, device)

        # 3. add predictions for all chromosomes
        add_predictions(df, cell_line=cell_line, model_cfg=model_cfg, chroms=list(range(1, 23)), genome=genome, margin_size=margin_size, window_size=window_size)
    else:
        raise ValueError("Model checkpoint not found.")
        # add_cv_predictions(task_config, df)

    # 4. save df
    print('Saving SNV predictions to', params.output_path)
    df.to_csv(path_or_buf=params.output_path, index=False)
