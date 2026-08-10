from __future__ import annotations

from typing import Dict, Any, Optional

import torch
import torch.nn as nn

from prmae.models.pinn_blocks import ResidualPINN, PhysicsAwareAttention


class PRMAEModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        window_sizes: Dict[str, int],  # {"micro": 6, "meso": 24, "macro": 144}
        attention_context_dim: int,
        temperature: float = 1.0,
        hidden_channels: int = 32,
    ):
        super().__init__()
        self.window_sizes = window_sizes
        self.micro = ResidualPINN(input_dim, window_sizes["micro"], hidden_channels=hidden_channels, kernel_size=3, dilation=1)
        self.meso = ResidualPINN(input_dim, window_sizes["meso"], hidden_channels=hidden_channels, kernel_size=3, dilation=4)
        self.macro = ResidualPINN(input_dim, window_sizes["macro"], hidden_channels=hidden_channels, kernel_size=5, dilation=8)

        # components: 3 PINNs + N specialists (we'll start with 1 specialist placeholder)
        self.num_components = 4
        self.attn = PhysicsAwareAttention(attention_context_dim, self.num_components, hidden_dim=64, temperature=temperature)

        # Linear head to map specialist feature to output (placeholder for CatBoost integration)
        self.specialist_head = nn.Linear(1, 1)

    def forward(
        self,
        x_micro: torch.Tensor,
        x_meso: torch.Tensor,
        x_macro: torch.Tensor,
        p_theoretical: torch.Tensor,
        attn_context: torch.Tensor,
        specialist_output: Optional[torch.Tensor] = None,
        attn_weights_override: Optional[torch.Tensor] = None,  # (B, 4)
        component_mask: Optional[torch.Tensor] = None,  # (B, 4) or (4,)
    ) -> Dict[str, torch.Tensor]:
        # x_*: (B, T_k, D), p_theoretical: (B,), attn_context: (B, F)
        r_micro_seq = self.micro(x_micro)  # (B, T_micro)
        r_meso_seq = self.meso(x_meso)
        r_macro_seq = self.macro(x_macro)

        # take last step residual from each window as aligned to prediction time
        r_micro = r_micro_seq[:, -1]
        r_meso = r_meso_seq[:, -1]
        r_macro = r_macro_seq[:, -1]

        if specialist_output is None:
            sp = self.specialist_head(torch.ones((x_micro.size(0), 1), device=x_micro.device))[:, 0]
        else:
            sp = specialist_output[:, 0]

        # Attention weights over [micro, meso, macro, specialist]
        if attn_weights_override is not None:
            weights = attn_weights_override
        else:
            weights = self.attn(attn_context)  # (B, 4)

        # Optional component mask (e.g., for ablations). Renormalize to sum=1.
        if component_mask is not None:
            if component_mask.dim() == 1:
                component_mask = component_mask.unsqueeze(0).expand(weights.size(0), -1)
            weights = weights * component_mask
            sums = weights.sum(dim=-1, keepdim=True)
            # If all masked -> fall back to uniform over available (avoid div-by-zero)
            zero_mask = (sums <= 1e-8)
            if zero_mask.any():
                # uniform over unmasked components
                mask = component_mask
                denom = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
                uni = mask / denom
                weights = torch.where(zero_mask, uni, weights)
                sums = weights.sum(dim=-1, keepdim=True)
            weights = weights / sums
        alpha_micro, alpha_meso, alpha_macro, beta_sp = torch.split(weights, 1, dim=-1)
        alpha_micro = alpha_micro[:, 0]
        alpha_meso = alpha_meso[:, 0]
        alpha_macro = alpha_macro[:, 0]
        beta_sp = beta_sp[:, 0]

        p_pred = p_theoretical + alpha_micro * r_micro + alpha_meso * r_meso + alpha_macro * r_macro + beta_sp * sp

        outputs = {
            "p_pred": p_pred,
            "r_micro": r_micro,
            "r_meso": r_meso,
            "r_macro": r_macro,
            "attn_weights": weights,
            "sp": sp,
        }
        return outputs
