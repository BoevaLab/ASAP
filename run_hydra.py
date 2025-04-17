# Example of running locally, with some adjusted parameters
# NB! main thing is paths
# python run_hydra.py run_script=train model=lstm experiment_name=test +cell_line=GM12878 train.fold=0 train.trainer.max_epochs=1 paths=local

import hydra

from src.config import TaskConfig
from src.tasks import train, eval_robustness, train_all, predict_snv, eval, export_predictions#, print_percentiles, generate


@hydra.main(version_base='1.3', config_path='conf', config_name='config')
def main(cfg):
    task_config = TaskConfig(cfg)

    if cfg.run_script == 'train':
        train.run(task_config)
    if cfg.run_script == 'train_all':
        train_all.run(task_config)
    elif cfg.run_script == 'predict_snv':
        predict_snv.run(task_config)
    elif cfg.run_script == 'eval':
        eval.run(task_config)
    elif cfg.run_script == 'export_predictions':
        export_predictions.run(task_config)
    elif cfg.run_script == 'eval_robustness':
        eval_robustness.run(task_config)
    else:
        print(f'Not a valid script name: {cfg.run_script}')


if __name__ == '__main__':
    main()
