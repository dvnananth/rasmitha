from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from prmae.data.loaders import load_t1_csv, load_texas_csv, compute_theoretical_power_from_curve
from prmae.data.power_curve import PowerCurve
from prmae.training.datamodules import create_dataloaders, WindowConfig, extract_specialist_arrays
from prmae.models.prmae_model import PRMAEModel
from prmae.models.catboost_specialist import CatBoostSpecialist, CatBoostConfig
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

    # CatBoost flags
    p.add_argument('--cb.iter', type=int, default=500)
    p.add_argument('--cb.depth', type=int, default=6)
    p.add_argument('--cb.lr', type=float, default=0.05)
    p.add_argument('--cb.l2', type=float, default=3.0)
    return p


def main():
    args = build_argparser().parse_args()

    pc = PowerCurve.from_ge_csv(args.power_curve_path)

    if args.t1_path:
        df = load_t1_csv(args.t1_path)
        df['power_theoretical'] = compute_theoretical_power_from_curve(df, pc)
        feature_columns = ['wind_speed', 'power_theoretical', 'wind_direction_deg', 'ti_proxy', 'yaw_var']
    elif args.texas_path:
        df = load_texas_csv(args.texas_path)
        df['power_theoretical'] = compute_theoretical_power_from_curve(df, pc)
        feature_columns = ['wind_speed', 'power_theoretical', 'wind_direction_deg', 'pressure_atm', 'air_temp_c', 'ti_proxy', 'yaw_var']
    else:
        raise ValueError('Provide --data.t1 or --data.texas')

    window_cfg = WindowConfig(micro=args.micro, meso=args.meso, macro=args.macro)
    train_loader, val_loader, test_loader, (train_ds, val_ds, test_ds) = create_dataloaders(
        df, feature_columns, batch_size=args.batch_size, window_config=window_cfg
    )

    # Train CatBoost on residuals r_total with context features
    cb_cfg = CatBoostConfig(depth=args.cb.depth, learning_rate=args.cb.lr, l2_leaf_reg=args.cb.l2, iterations=args.cb.iter)
    cb = CatBoostSpecialist(cb_cfg)
    sp_arrays = extract_specialist_arrays(train_ds, val_ds, test_ds)
    cb.fit(sp_arrays.X_train, sp_arrays.y_train, sp_arrays.X_val, sp_arrays.y_val)

    input_dim = len(feature_columns)
    attn_context_dim = input_dim

    model = PRMAEModel(input_dim=input_dim, window_sizes={"micro": args.micro, "meso": args.meso, "macro": args.macro}, attention_context_dim=attn_context_dim)
    trainer = PRMAETrainer(model, physics=PhysicsParams(), weights=LossWeights(), lr=args.lr, device=args.device)

    for epoch in range(args.epochs):
        total = 0.0
        count = 0
        for batch in train_loader:
            # get specialist preds for this batch
            sp = cb.predict(batch['attn_context'].numpy())  # (B,1)
            batch = {k: v for k, v in batch.items()}
            batch['specialist_output'] = torch.from_numpy(sp).to(args.device)
            # forward with specialist
            trainer.model.train()
            outputs = trainer.model(
                batch['x_micro'].to(args.device),
                batch['x_meso'].to(args.device),
                batch['x_macro'].to(args.device),
                batch['p_theoretical'].to(args.device),
                batch['attn_context'].to(args.device),
                batch['specialist_output'],
            )
            # compute loss using trainer logic (reuse step without double forward)
            # Rebuild batch dict expected by trainer
            tbatch = {
                'x_micro': batch['x_micro'],
                'x_meso': batch['x_meso'],
                'x_macro': batch['x_macro'],
                'p_theoretical': batch['p_theoretical'],
                'y': batch['y'],
                'attn_context': batch['attn_context'],
                'r_total': batch['r_total'],
            }
            # Manual loss since trainer.step_batch does forward
            p_pred = outputs['p_pred']
            r_micro = outputs['r_micro']
            r_meso = outputs['r_meso']
            r_macro = outputs['r_macro']
            attn_w = outputs['attn_weights']

            from prmae.losses.losses import rmse_loss, pde_local_energy_loss, boundary_physical_range_loss, scale_consistency_loss, phys_att_regularizer
            y = tbatch['y'].to(args.device)
            p_theoretical = tbatch['p_theoretical'].to(args.device)
            wind_speed = tbatch['x_micro'][:, -1, 0].to(args.device)
            denom = 0.5 * 1.225 * 5026.5 * (wind_speed ** 3 + 1e-6)
            cp_est = torch.clamp(p_pred / denom, 0.0, 1.0)
            l_data = rmse_loss(p_pred, y)
            l_pde = pde_local_energy_loss(p_pred, 1.225, 5026.5, cp_est, wind_speed)
            l_bc = boundary_physical_range_loss(cp_est)
            l_scale = scale_consistency_loss(r_micro, r_meso, r_macro, tbatch['r_total'].to(args.device))
            phys_incons = torch.tanh(torch.abs(y - p_theoretical) / (torch.abs(p_theoretical) + 1e-3))
            l_physatt = phys_att_regularizer(attn_w, phys_incons)
            loss = l_data + 0.3 * l_pde + 0.1 * l_bc + 0.05 * l_scale + 0.01 * l_physatt

            trainer.opt.zero_grad()
            loss.backward()
            trainer.opt.step()

            total += float(loss.detach().cpu())
            count += 1
        print(f"Epoch {epoch+1}/{args.epochs} train_loss={total/max(1,count):.4f}")

    # quick eval on test
    model.eval()
    ys = []
    ps = []
    with torch.no_grad():
        for batch in test_loader:
            sp = cb.predict(batch['attn_context'].numpy())
            out = model(
                batch['x_micro'].to(args.device),
                batch['x_meso'].to(args.device),
                batch['x_macro'].to(args.device),
                batch['p_theoretical'].to(args.device),
                batch['attn_context'].to(args.device),
                torch.from_numpy(sp).to(args.device),
            )
            ys.append(batch['y'].numpy())
            ps.append(out['p_pred'].cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    rmse = float(np.sqrt(np.mean((y-p)**2)))
    print(f"Test RMSE: {rmse:.3f} kW")


if __name__ == '__main__':
    main()
