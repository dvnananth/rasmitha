from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import Pool

from prmae.data.loaders import load_t1_csv, load_texas_csv, compute_theoretical_power_from_curve
from prmae.data.power_curve import PowerCurve
from prmae.evaluation.metrics import r2_score, mae, rmse, physics_validity_score
from prmae.models.catboost_specialist import CatBoostSpecialist, CatBoostConfig


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PR-MAE analytics: CatBoost-only reporting with metrics and plots")
    p.add_argument('--data.t1', dest='t1_path', type=str, required=False)
    p.add_argument('--data.texas', dest='texas_path', type=str, required=False)
    p.add_argument('--data.power_curve', dest='power_curve_path', type=str, required=True)

    p.add_argument('--output_dir', type=str, default=None)
    p.add_argument('--cv.folds', dest='cv_folds', type=int, default=1, help='Number of blocked CV folds (1=holdout)')

    # Physics constants
    p.add_argument('--rho', type=float, default=1.225, help='Air density (kg/m^3)')
    p.add_argument('--rotor_area', type=float, default=5026.5, help='Rotor swept area (m^2)')

    # CatBoost
    p.add_argument('--cb.iter', type=int, default=800)
    p.add_argument('--cb.depth', type=int, default=8)
    p.add_argument('--cb.lr', type=float, default=0.03)
    p.add_argument('--cb.l2', type=float, default=5.0)

    # Optional analyses
    p.add_argument('--shap', action='store_true', help='Compute SHAP attributions (can be slow)')
    return p


def prepare_dataframe(args: argparse.Namespace) -> Tuple[pd.DataFrame, List[str]]:
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
    return df, feature_columns


def blocked_folds_indices(n: int, k: int) -> List[Tuple[slice, slice]]:
    """Return list of (train_slice, test_slice) for blocked CV.
    train is [0:start), test is [start:end). Skip folds where train is empty.
    """
    folds = []
    for i in range(k):
        start = int(n * i / k)
        end = int(n * (i + 1) / k)
        if start == 0:
            # skip fold with empty train
            continue
        folds.append((slice(0, start), slice(start, end)))
    if not folds:
        # fallback to 70/15/15 holdout
        i_train = int(n * 0.7)
        i_val = int(n * 0.85)
        folds = [(slice(0, i_val), slice(i_val, n))]
    return folds


def ensure_outdir(base: str | None) -> str:
    if base is None:
        base = os.path.join(os.getcwd(), 'reports', datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(base, exist_ok=True)
    return base


def plot_timeseries(y_true: np.ndarray, y_pred: np.ndarray, path: str) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(y_true, label='Observed', linewidth=1.5)
    plt.plot(y_pred, label='Predicted', linewidth=1.2)
    plt.xlabel('Time index')
    plt.ylabel('Power (kW)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, path: str) -> None:
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x=y_true, y=y_pred, s=10, alpha=0.5)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--', linewidth=1)
    plt.xlabel('Observed (kW)')
    plt.ylabel('Predicted (kW)')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_residual_vs_ws(residual: np.ndarray, ws: np.ndarray, path: str) -> None:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=ws, y=residual, s=10, alpha=0.4)
    plt.axhline(0.0, color='k', linestyle='--', linewidth=1)
    plt.xlabel('Wind speed (m/s)')
    plt.ylabel('Residual (kW)')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_error_hist(residual: np.ndarray, path: str) -> None:
    plt.figure(figsize=(6, 4))
    sns.histplot(residual, bins=50, kde=True)
    plt.xlabel('Residual (kW)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_feature_importance(model: CatBoostSpecialist, feature_names: List[str], path: str) -> None:
    importances = model.model.get_feature_importance()
    order = np.argsort(importances)[::-1]
    plt.figure(figsize=(6, 4))
    sns.barplot(x=importances[order], y=np.array(feature_names)[order])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_shap_summary(model: CatBoostSpecialist, X: np.ndarray, feature_names: List[str], path: str) -> None:
    pool = Pool(X, feature_names=feature_names)
    shap_vals = model.model.get_feature_importance(pool, type='ShapValues')
    # Last column is expected value; drop it
    shap_vals = shap_vals[:, :-1]
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    order = np.argsort(mean_abs)[::-1]
    plt.figure(figsize=(6, 4))
    sns.barplot(x=mean_abs[order], y=np.array(feature_names)[order])
    plt.xlabel('Mean |SHAP|')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    args = build_argparser().parse_args()
    outdir = ensure_outdir(args.output_dir)

    df, feature_columns = prepare_dataframe(args)
    X = df[feature_columns].values.astype(np.float32)
    power_kw = df['power_kw'].values.astype(np.float32)
    pth = df['power_theoretical'].values.astype(np.float32)
    r_total = (power_kw - pth).astype(np.float32)
    ws = df['wind_speed'].values.astype(np.float32)

    folds = blocked_folds_indices(len(df), args.cv_folds)

    metrics_all = []
    preds_all = []

    for fi, (tr, te) in enumerate(folds):
        fold_dir = os.path.join(outdir, f'fold_{fi+1}')
        os.makedirs(fold_dir, exist_ok=True)

        X_train, y_train = X[tr], r_total[tr]
        X_test, y_test = X[te], r_total[te]
        pth_test = pth[te]
        ws_test = ws[te]
        y_true = pth_test + y_test

        # simple small validation from tail of train
        if len(X_train) > 10:
            i_val = int(len(X_train) * 0.9)
            X_tr, y_tr = X_train[:i_val], y_train[:i_val]
            X_val, y_val = X_train[i_val:], y_train[i_val:]
        else:
            X_tr, y_tr = X_train, y_train
            X_val, y_val = None, None

        cb_cfg = CatBoostConfig(depth=args.cb.depth, learning_rate=args.cb.lr, l2_leaf_reg=args.cb.l2, iterations=args.cb.iter)
        cb = CatBoostSpecialist(cb_cfg)
        cb.fit(X_tr, y_tr, X_val, y_val)

        r_pred = cb.predict(X_test).reshape(-1)
        p_pred = pth_test + r_pred

        fold_metrics = {
            'rmse': rmse(y_true, p_pred),
            'mae': mae(y_true, p_pred),
            'r2': r2_score(y_true, p_pred),
        }
        # Physics validity
        denom = 0.5 * args.rho * args.rotor_area * (np.power(ws_test, 3) + 1e-6)
        cp_est = np.clip(p_pred / denom, 0.0, 1.0)
        fold_metrics['physics_validity'] = physics_validity_score(cp_est)

        metrics_all.append(fold_metrics)
        preds_all.append({
            'y_true': y_true.tolist(),
            'p_pred': p_pred.tolist(),
            'pth': pth_test.tolist(),
        })

        # Plots
        plot_timeseries(y_true, p_pred, os.path.join(fold_dir, 'timeseries.png'))
        plot_scatter(y_true, p_pred, os.path.join(fold_dir, 'scatter_obs_pred.png'))
        plot_residual_vs_ws(p_pred - y_true, ws_test, os.path.join(fold_dir, 'residual_vs_ws.png'))
        plot_error_hist(p_pred - y_true, os.path.join(fold_dir, 'residual_hist.png'))
        plot_feature_importance(cb, feature_columns, os.path.join(fold_dir, 'feature_importance.png'))
        if args.shap:
            plot_shap_summary(cb, X_test, feature_columns, os.path.join(fold_dir, 'shap_summary.png'))

    # Save metrics summary
    summary = {
        'args': vars(args),
        'fold_metrics': metrics_all,
        'aggregate': {
            'rmse_mean': float(np.mean([m['rmse'] for m in metrics_all])),
            'mae_mean': float(np.mean([m['mae'] for m in metrics_all])),
            'r2_mean': float(np.mean([m['r2'] for m in metrics_all])),
            'physics_validity_mean': float(np.mean([m['physics_validity'] for m in metrics_all])),
        },
    }
    with open(os.path.join(outdir, 'metrics.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Report saved to: {outdir}")


if __name__ == '__main__':
    main()
