import os

from src.config import CONFIG


def get_checkpoint_for_fold(checkpoint_dir: str, exp: str, model_name: str, cell_line: str, fold: int):
    all_files = os.listdir(checkpoint_dir)
    should_contain = [exp, f'_{model_name}_', cell_line, f'fold{fold}']
    matches = list(filter(lambda f: all(s in f for s in should_contain), all_files))
    # assert len(matches) == 1, f'Did not get exactly one match for ' \
    #                           f'{checkpoint_dir}/{exp}_{model_name}_{cell_line}_fold{fold}\nMatches:\n\t{matches}'
    if len(matches) != 1:
        print(f'Did not get exactly one match for '
              f'{checkpoint_dir}/{exp}_{model_name}_{cell_line}_fold{fold}\nMatches:\n\t{matches}')
        return None
    return f'{checkpoint_dir}/{matches[0]}'


def get_checkpoints_all_folds(checkpoint_dir: str, train_exp: str, model_name: str, cell_line_exp: str):
    return [get_checkpoint_for_fold(checkpoint_dir, train_exp, model_name, cell_line_exp, fold) for fold in
            range(CONFIG.nr_folds)]


def get_checkpoint_all_chroms(checkpoint_dir: str, train_exp: str, model_name: str, cell_line_exp: str):
    all_files = os.listdir(checkpoint_dir)
    should_contain = [train_exp, cell_line_exp]
    matches = list(filter(lambda f: all(s in f for s in should_contain), all_files))
    if len(matches) > 1:
        matches = [x for x in matches if 'dcnn' not in x]
    if len(matches) != 1:
        print(f'Did not get exactly one match for '
              f'{checkpoint_dir}/{train_exp}_*\nMatches:\n\t{matches}')
        return None
    return f'{checkpoint_dir}/{train_exp}'
