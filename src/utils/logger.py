import pathlib
import torch
import logging
logging.basicConfig(level=logging.INFO)

try:
    import wandb
    WANDB_LOGGER_AVAILABLE = True
except ImportError:
    WANDB_LOGGER_AVAILABLE = False
    print("WandB is not installed. WandBLogger not available. Using TextLogger instead.")


def get_default_logger(logs_dir, **kwargs):
    if WANDB_LOGGER_AVAILABLE:
        return WandBLogger(run=kwargs['run_cfg'], logs_dir=logs_dir)
    else:
        return TextLogger(logs_dir=logs_dir)

class Logger():
    def __init__(self, **kwargs) -> None:
        pass

    def log(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def save_model(self, model, name) -> None:
        raise NotImplementedError


class WandBLogger(Logger):
    def __init__(self, run, logs_dir='logs/') -> None:
        super().__init__()
        assert WANDB_LOGGER_AVAILABLE, "WandB is not installed. WandBLogger not available."
        if isinstance(run, dict):
            wandb_cfg = {
                'project': 'mutiger-snv-paper',
                'entity': 'sdsc-paired-hydro',
            }
            wandb_cfg['config'] = run
            self.run: wandb.wandb_sdk.wandb_run.Run = wandb.init(**wandb_cfg)
        else:
            self.run: wandb.wandb_sdk.wandb_run.Run = run
        self.logs_dir = logs_dir

    def log(self, *args, step=None, **kwargs,) -> None:
        self.run.log(*args, **kwargs)

    def save_model(self, model, name) -> None:
        path = pathlib.Path(f"{self.logs_dir}/{name}/")
        path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path / "checkpoint.pth")
        artifact = wandb.Artifact(self.run.name +'_best_corr_model', type='model')
        artifact.add_file(path / "checkpoint.pth")
        wandb.run.log_artifact(artifact)

class TextLogger(Logger):
    def __init__(self, logs_dir='logs') -> None:
        super().__init__()
        self.logger = logging.getLogger("run")
        self.logs_dir = logs_dir

    def log(self, *args, **kwargs) -> None:
        msg = ""
        score_info = args[0]
        for k, val in kwargs.items():
            msg += f"{k}: {val} \n"
        for k, val in score_info.items():
            msg += f"{k}: {val} ; "
        self.logger.info(msg)

    def save_model(self, model, name) -> None:
        path = pathlib.Path(f"{self.logs_dir}/{name}/")
        path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path / "checkpoint.pth")
