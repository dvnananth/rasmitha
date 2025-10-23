from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from prmae.data.loaders import load_t1_csv, load_texas_csv, compute_theoretical_power_from_curve
from prmae.data.power_curve import PowerCurve
from prmae.training.datamodules import create_dataloaders, WindowConfig
from prmae.models.prmae_model import PRMAEModel
from prmae.training.trainers import PRMAETrainer, PhysicsParams, LossWeights


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train PR-MAE model")
    p.add_argument('--data.t1', dest='t1_path', type=str, required=False)
    p.add_argument('--data.power_curve', dest='power_curve_path', type=str, required=True)
    p.add_argument('--data.texas', dest='texas_path', type=str, required=False)

    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--lr', type=float, default=1e-3)

    p.add_argument('--micro', type=int, default=6)
    p.add_argument('--meso', type=int, default=24)
    p.add_argument('--macro', type=int, default=144)

    p.add_argument('--device', type=str, default='cpu')
    return p


def main():
    args = build_argparser().parse_args()

    pc = PowerCurve.from_ge_csv(args.power_curve_path)

    if args.t1_path:
        df = load_t1_csv(args.t1_path)
        # if theoretical not trustworthy, recompute from curve
        df['power_theoretical'] = compute_theoretical_power_from_curve(df, pc)
        feature_columns = ['wind_speed', 'power_theoretical', 'wind_direction_deg', 'ti_proxy', 'yaw_var']
    elif args.texas_path:
        df = load_texas_csv(args.texas_path)
        df['power_theoretical'] = compute_theoretical_power_from_curve(df, pc)
        feature_columns = ['wind_speed', 'power_theoretical', 'wind_direction_deg', 'pressure_atm', 'air_temp_c', 'ti_proxy', 'yaw_var']
    else:
        raise ValueError('Provide --data.t1 or --data.texas')

    window_cfg = WindowConfig(micro=args.micro, meso=args.meso, macro=args.macro)
    train_loader, val_loader, test_loader = create_dataloaders(df, feature_columns, batch_size=args.batch_size, window_config=window_cfg)

    input_dim = len(feature_columns)
    attn_context_dim = input_dim

    model = PRMAEModel(input_dim=input_dim, window_sizes={"micro": args.micro, "meso": args.meso, "macro": args.macro}, attention_context_dim=attn_context_dim)
    trainer = PRMAETrainer(model, physics=PhysicsParams(), weights=LossWeights(), lr=args.lr, device=args.device)

    for epoch in range(args.epochs):
        total = 0.0
        count = 0
        for batch in train_loader:
            metrics = trainer.step_batch(batch)
            total += metrics['loss']
            count += 1
        print(f"Epoch {epoch+1}/{args.epochs} train_loss={total/max(1,count):.4f}")

    # quick eval on test
    model.eval()
    ys = []
    ps = []
    with torch.no_grad():
        for batch in test_loader:
            out = model(
                batch['x_micro'].to(args.device),
                batch['x_meso'].to(args.device),
                batch['x_macro'].to(args.device),
                batch['p_theoretical'].to(args.device),
                batch['attn_context'].to(args.device),
            )
            ys.append(batch['y'].numpy())
            ps.append(out['p_pred'].cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    rmse = float(np.sqrt(np.mean((y-p)**2)))
    print(f"Test RMSE: {rmse:.3f} kW")


if __name__ == '__main__':
    main()
