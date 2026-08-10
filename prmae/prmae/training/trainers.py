from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import Adam

from prmae.models.prmae_model import PRMAEModel
from prmae.losses.losses import rmse_loss, pde_local_energy_loss, boundary_physical_range_loss, scale_consistency_loss, phys_att_regularizer


@dataclass
class PhysicsParams:
    air_density: float = 1.225  # kg/m^3
    rotor_area: float = 5026.5  # example for ~80 m rotor radius; adjust via CLI


@dataclass
class LossWeights:
    lambda_pde: float = 0.3
    lambda_bc: float = 0.1
    lambda_scale: float = 0.05
    lambda_physatt: float = 0.01


class PRMAETrainer:
    def __init__(self, model: PRMAEModel, physics: PhysicsParams = PhysicsParams(), weights: LossWeights = LossWeights(), lr: float = 1e-3, device: str = "cpu"):
        self.model = model.to(device)
        self.physics = physics
        self.weights = weights
        self.device = device
        self.opt = Adam(self.model.parameters(), lr=lr)

    def step_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        x_micro = batch["x_micro"].to(self.device)
        x_meso = batch["x_meso"].to(self.device)
        x_macro = batch["x_macro"].to(self.device)
        p_theoretical = batch["p_theoretical"].to(self.device)
        y = batch["y"].to(self.device)
        attn_context = batch["attn_context"].to(self.device)
        r_total = batch["r_total"].to(self.device)

        outputs = self.model(x_micro, x_meso, x_macro, p_theoretical, attn_context)
        p_pred = outputs["p_pred"]
        r_micro = outputs["r_micro"]
        r_meso = outputs["r_meso"]
        r_macro = outputs["r_macro"]
        attn_w = outputs["attn_weights"]

        # Data loss
        l_data = rmse_loss(p_pred, y)

        # Simple Cp estimate from predicted vs wind speed proxy; here approximate cp_est = p_pred / (0.5 rho A v^3) clipped
        wind_speed = x_micro[:, -1, 0]  # assume first feature is wind_speed
        denom = 0.5 * self.physics.air_density * self.physics.rotor_area * (wind_speed ** 3 + 1e-6)
        cp_est = torch.clamp(p_pred / denom, 0.0, 1.0)

        l_pde = pde_local_energy_loss(p_pred, self.physics.air_density, self.physics.rotor_area, cp_est, wind_speed)
        l_bc = boundary_physical_range_loss(cp_est)
        l_scale = scale_consistency_loss(r_micro, r_meso, r_macro, r_total)

        # physics inconsistency proxy: normalized absolute residual wrt theoretical
        phys_incons = torch.tanh(torch.abs(y - p_theoretical) / (torch.abs(p_theoretical) + 1e-3))
        l_physatt = phys_att_regularizer(attn_w, phys_incons)

        loss = l_data + self.weights.lambda_pde * l_pde + self.weights.lambda_bc * l_bc + self.weights.lambda_scale * l_scale + self.weights.lambda_physatt * l_physatt

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return {
            "loss": float(loss.detach().cpu()),
            "l_data": float(l_data.detach().cpu()),
            "l_pde": float(l_pde.detach().cpu()),
            "l_bc": float(l_bc.detach().cpu()),
            "l_scale": float(l_scale.detach().cpu()),
            "l_physatt": float(l_physatt.detach().cpu()),
        }
