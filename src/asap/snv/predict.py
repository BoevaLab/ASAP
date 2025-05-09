import json
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from asap.dataloader.bw_to_data import get_data_by_idx
from asap.dataloader.utils.seq import seq_to_idx

def _idx_to_seq(idx: np.ndarray) -> np.ndarray:
    seq = np.array(["A", "G", "C", "T", "N"])[idx]
    return "".join(seq)

def _idx_to_ohe(idx: np.ndarray) -> np.ndarray:
    eye = np.concatenate(
        (np.eye(4, dtype=idx.dtype), np.zeros((1, 4), dtype=idx.dtype)), axis=0
    )
    one_hot_encoded = eye[idx]
    return one_hot_encoded

def add_predictions(snv: pd.DataFrame, chroms: List[int], bw_file: str, model: nn.Module, genome: str, margin_size: int, window_size: int, bin_size: int, device: torch.device) -> None:
    # Ensure output columns exist
    output_columns = ['signal_true', 'signal_pred_ref', 'signal_pred_alt']
    for col in output_columns:
        if col not in snv.columns:
            snv[col] = None

    for chrom in chroms:
        chr_df = snv[(snv.chr == f'chr{chrom}') & (snv[output_columns].isnull().any(axis=1))]

        n_samples = len(chr_df)
        if n_samples == 0:
            print(f'No SNVs to predict in chr{chrom}')
            continue
        
        print(f'Adding predictions for {n_samples} SNVs in chr{chrom}')

        start_idx = 0
        while start_idx < n_samples:
            end_idx = min(start_idx + 128, n_samples)
            chr_df_i = chr_df.iloc[[*range(start_idx, end_idx)]]
            starts = chr_df_i.pos.to_numpy() - window_size// 2
            x, y = get_data_by_idx(signal_files=[bw_file], chrom=chrom, bin_size=bin_size,
                                window=window_size,
                                seq_starts=starts, genome=genome, margin=margin_size)

            x_var = x.copy()
            mid_loc = (window_size + (2*margin_size)) // 2 - 1
            for i, (_, row) in enumerate(chr_df_i.iterrows()):
                if row.ref != _idx_to_seq(x[i][mid_loc]):
                    print(f'Incorrect reference for entry {i}; pos {row.pos}; from vcf: ref {row.ref}, {row.alt}; from fasta: {_idx_to_seq(x[i][mid_loc-2:mid_loc+3])}')
                x_var[i][mid_loc] = seq_to_idx(row.alt)
            
            x = _idx_to_ohe(x).astype(np.float32)
            x_var = _idx_to_ohe(x_var).astype(np.float32)

            ref_pred = model(torch.from_numpy(x).to(device))
            var_pred = model(torch.from_numpy(x_var).to(device))
            
            for (i, _), true, ref, var in zip(chr_df_i.iterrows(), y, ref_pred.cpu().detach().numpy(), var_pred.cpu().detach().numpy()):
                snv.loc[i, 'signal_true'] = json.dumps(true.flatten().tolist())
                snv.loc[i, 'signal_pred_ref'] = json.dumps(ref.tolist())
                snv.loc[i, 'signal_pred_alt'] = json.dumps(var.tolist())
            
            start_idx = end_idx
    return snv
