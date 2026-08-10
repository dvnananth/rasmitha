from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


BETZ_LIMIT = 0.593


def rmse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(F.mse_loss(y_pred, y_true))


def pde_local_energy_loss(p_pred: torch.Tensor, rho: float, area: float, cp_est: torch.Tensor, wind_speed: torch.Tensor) -> torch.Tensor:
    # Encourage P ≈ 0.5 * rho * A * Cp * v^3 with Cp within [0, BETZ]
    p_phys = 0.5 * rho * area * cp_est * (wind_speed ** 3)
    return F.l1_loss(p_pred, p_phys)


def boundary_physical_range_loss(cp_est: torch.Tensor) -> torch.Tensor:
    # Penalize Cp > Betz or Cp < 0
    over = F.relu(cp_est - BETZ_LIMIT)
    under = F.relu(-cp_est)
    return (over + under).mean()


def scale_consistency_loss(r_micro: torch.Tensor, r_meso: torch.Tensor, r_macro: torch.Tensor, r_total: torch.Tensor) -> torch.Tensor:
    # L2 consistency: sum of scales should match total residual
    sum_r = r_micro + r_meso + r_macro
    return F.mse_loss(sum_r, r_total)


def phys_att_regularizer(attn_weights: torch.Tensor, phys_inconsistency: torch.Tensor, pin_idx: slice = slice(0, 3)) -> torch.Tensor:
    # Encourage higher PINN weights when physics inconsistency is high
    # attn_weights: (B, C), phys_inconsistency: (B,) in [0,1]
    pinn_sum = attn_weights[:, pin_idx].sum(dim=-1)
    target = phys_inconsistency  # larger -> want larger pinn_sum
    return F.mse_loss(pinn_sum, target)
