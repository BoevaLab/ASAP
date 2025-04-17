import json

import numpy as np
import pandas as pd
import torch

from src.config import CellLineConfig
from src.config import ModelConfig
from src.dataloader.bw_to_data import get_data_by_idx
from src.dataloader.utils.seq import seq_to_idx

def idx_to_seq(idx: np.ndarray) -> np.ndarray:
    seq = np.array(["A", "G", "C", "T", "N"])[idx]
    return "".join(seq)

def idx_to_ohe(idx: np.ndarray) -> np.ndarray:
    eye = np.concatenate(
        (np.eye(4, dtype=idx.dtype), np.zeros((1, 4), dtype=idx.dtype)), axis=0
    )
    one_hot_encoded = eye[idx]
    return one_hot_encoded

def add_predictions(df: pd.DataFrame, chroms: list[int], cell_line: CellLineConfig, model_cfg: ModelConfig, genome: str, margin_size: int, window_size: int):
    # Ensure output columns exist
    output_columns = ['signal_true', 'signal_pred_ref', 'signal_pred_alt']
    for col in output_columns:
        if col not in df.columns:
            df[col] = None

    batch_size = 128

    for chrom in chroms:
        chr_df = df[(df.chr == f'chr{chrom}') & (df[output_columns].isnull().any(axis=1))]
        print(f'Adding predictions for {len(chr_df)} SNVs in chr{chrom}')

        n_samples = len(chr_df)
        if n_samples == 0:
            continue
        
        start_idx = 0
        while start_idx < n_samples:
            end_idx = min(start_idx + 128, n_samples)
            chr_df_i = chr_df.iloc[[*range(start_idx, end_idx)]]
            starts = chr_df_i.pos.to_numpy() - window_size// 2
            x, y = get_data_by_idx(signal_files=[cell_line.bw], chrom=chrom, bin_size=model_cfg.bin_size,
                                window=window_size,
                                seq_starts=starts, genome=genome, margin=margin_size)

            x_var = x.copy()
            mid_loc = (window_size + (2*margin_size)) // 2 - 1
            for i, (_, row) in enumerate(chr_df_i.iterrows()):
                if row.ref != idx_to_seq(x[i][mid_loc]):
                    print(f'!!!!!!! {i} {row.pos}, {row.ref}, {row.alt}, {idx_to_seq(x[i][mid_loc-2:mid_loc+3])}')
                x_var[i][mid_loc] = seq_to_idx(row.alt)
            
            x = idx_to_ohe(x).astype(np.float32)
            x_var = idx_to_ohe(x_var).astype(np.float32)

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            ref_pred = model_cfg.model(torch.from_numpy(x).to(device))
            var_pred = model_cfg.model(torch.from_numpy(x_var).to(device))
            
            for (i, _), true, ref, var in zip(chr_df_i.iterrows(), y, ref_pred.cpu().detach().numpy(), var_pred.cpu().detach().numpy()):
                df.loc[i, 'signal_true'] = json.dumps(true.flatten().tolist())
                df.loc[i, 'signal_pred_ref'] = json.dumps(ref.tolist())
                df.loc[i, 'signal_pred_alt'] = json.dumps(var.tolist())
            
            start_idx = end_idx
