from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from prmae.utils.common import fit_minmax_scalers, apply_scalers, make_sliding_windows


@dataclass
class WindowConfig:
    micro: int = 6
    meso: int = 24
    macro: int = 144


class PRMAEDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list,
        target_column: str = "power_kw",
        window_config: WindowConfig = WindowConfig(),
        step: int = 1,
    ):
        self.df = df.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.window_config = window_config
        self.step = step

        X = self.df[self.feature_columns].values.astype(np.float32)
        y = self.df[self.target_column].values.astype(np.float32)

        self.X_micro = make_sliding_windows(X, window_config.micro, step)
        self.X_meso = make_sliding_windows(X, window_config.meso, step)
        self.X_macro = make_sliding_windows(X, window_config.macro, step)
        # Align y to window ends (the shortest window determines count)
        n = self.X_micro.shape[0]
        self.y = y[window_config.micro - 1 : window_config.micro - 1 + n]
        self.p_theoretical = self.df["power_theoretical"].values.astype(np.float32)[
            window_config.micro - 1 : window_config.micro - 1 + n
        ]

        # Attn context uses latest step features from the micro window end
        self.attn_context = self.X_micro[:, -1, :]  # (N, D)

        # Total residual
        self.r_total = self.y - self.p_theoretical

    def __len__(self) -> int:
        return self.X_micro.shape[0]

    def __getitem__(self, idx: int):
        return {
            "x_micro": torch.from_numpy(self.X_micro[idx]),
            "x_meso": torch.from_numpy(self.X_meso[idx]),
            "x_macro": torch.from_numpy(self.X_macro[idx]),
            "p_theoretical": torch.tensor(self.p_theoretical[idx]),
            "y": torch.tensor(self.y[idx]),
            "attn_context": torch.from_numpy(self.attn_context[idx]),
            "r_total": torch.tensor(self.r_total[idx]),
        }


def create_dataloaders(
    df: pd.DataFrame,
    feature_columns: list,
    batch_size: int = 64,
    window_config: WindowConfig = WindowConfig(),
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    step: int = 1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    n = len(df)
    i_train = int(n * train_ratio)
    i_val = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:i_train]
    val_df = df.iloc[i_train:i_val]
    test_df = df.iloc[i_val:]

    train_ds = PRMAEDataset(train_df, feature_columns, window_config=window_config, step=step)
    val_ds = PRMAEDataset(val_df, feature_columns, window_config=window_config, step=step)
    test_ds = PRMAEDataset(test_df, feature_columns, window_config=window_config, step=step)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader
