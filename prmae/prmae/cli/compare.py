from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor, RANSACRegressor, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from prmae.data.loaders import load_t1_csv, load_texas_csv, compute_theoretical_power_from_curve
from prmae.data.power_curve import PowerCurve
from prmae.evaluation.metrics import r2_score, mae, rmse
from prmae.models.catboost_specialist import CatBoostSpecialist, CatBoostConfig


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Comparison: Physics-only vs AI-only vs Hybrid (+ baselines)")
    p.add_argument('--data.t1', dest='t1_path', type=str, required=False)
    p.add_argument('--data.texas', dest='texas_path', type=str, required=False)
    p.add_argument('--data.power_curve', dest='power_curve_path', type=str, required=True)

    p.add_argument('--output_dir', type=str, default=None)
    p.add_argument('--cv.folds', dest='cv_folds', type=int, default=1)
    p.add_argument('--include_baselines', action='store_true')

    # CatBoost config (use underscores when calling)
    p.add_argument('--cb_iter', type=int, default=800)
    p.add_argument('--cb_depth', type=int, default=8)
    p.add_argument('--cb_lr', type=float, default=0.03)
    p.add_argument('--cb_l2', type=float, default=5.0)

    return p


def ensure_outdir(base: str | None) -> str:
    if base is None:
        base = os.path.join(os.getcwd(), 'reports', f'compare_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(base, exist_ok=True)
    return base


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
    folds = []
    for i in range(k):
        start = int(n * i / k)
        end = int(n * (i + 1) / k)
        if start == 0:
            continue
        folds.append((slice(0, start), slice(start, end)))
    if not folds:
        i_train = int(n * 0.7)
        i_val = int(n * 0.85)
        folds = [(slice(0, i_val), slice(i_val, n))]
    return folds


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        'MAE': mae(y_true, y_pred),
        'MSE': float(np.mean((y_true - y_pred) ** 2)),
        'RMSE': rmse(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
    }


def fit_predict_catboost_residual(X_tr: np.ndarray, y_res_tr: np.ndarray, X_val: np.ndarray | None, y_res_val: np.ndarray | None, X_te: np.ndarray, pth_te: np.ndarray, cfg: CatBoostConfig) -> Tuple[np.ndarray, Dict[str, float]]:
    t0 = time.time()
    sp = CatBoostSpecialist(cfg)
    sp.fit(X_tr, y_res_tr, X_val, y_res_val)
    t_train = time.time() - t0
    t1 = time.time()
    r_pred = sp.predict(X_te).reshape(-1)
    y_pred = pth_te + r_pred
    t_test = time.time() - t1
    return y_pred, {'train_s': t_train, 'test_s': t_test}


def fit_predict_catboost_direct(X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray | None, y_val: np.ndarray | None, X_te: np.ndarray, cfg_args: Dict) -> Tuple[np.ndarray, Dict[str, float]]:
    model = CatBoostRegressor(
        depth=cfg_args['depth'],
        learning_rate=cfg_args['learning_rate'],
        l2_leaf_reg=cfg_args['l2_leaf_reg'],
        iterations=cfg_args['iterations'],
        loss_function='RMSE',
        verbose=0,
        random_seed=42,
    )
    t0 = time.time()
    if X_val is not None and y_val is not None:
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    else:
        model.fit(X_tr, y_tr)
    t_train = time.time() - t0
    t1 = time.time()
    y_pred = model.predict(X_te)
    t_test = time.time() - t1
    return y_pred, {'train_s': t_train, 'test_s': t_test}


def build_baselines() -> Dict[str, object]:
    return {
        'GB': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, loss='squared_error', random_state=42),
        'MLP': MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500, random_state=42),
        'Ridge': Ridge(alpha=1.0),
        'LR': LinearRegression(fit_intercept=True, positive=False),
        'BR': BayesianRidge(n_iter=300),
        'RF': RandomForestRegressor(n_estimators=100, bootstrap=True, random_state=42, n_jobs=-1),
        'RSC': RANSACRegressor(random_state=42, max_trials=100, min_samples=0.5, loss='absolute_error'),
        'KNN': KNeighborsRegressor(n_neighbors=5, weights='uniform'),
        'HR': HuberRegressor(alpha=0.0001, epsilon=1.35),
        'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
        'DTR': DecisionTreeRegressor(random_state=42),
    }


def fit_predict_sklearn(model, X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    t0 = time.time()
    model.fit(X_tr, y_tr)
    t_train = time.time() - t0
    t1 = time.time()
    y_pred = model.predict(X_te)
    t_test = time.time() - t1
    return y_pred, {'train_s': t_train, 'test_s': t_test}


def plot_bar_metrics(agg: Dict[str, Dict[str, float]], out_path: str) -> None:
    # agg: model -> metric -> value
    models = list(agg.keys())
    metrics = ['MAE', 'MSE', 'RMSE', 'R2']
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()
    for i, m in enumerate(metrics):
        vals = [agg[model][m] for model in models]
        sns.barplot(x=models, y=vals, ax=axes[i])
        axes[i].set_title(m)
        axes[i].tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    args = build_argparser().parse_args()
    outdir = ensure_outdir(args.output_dir)

    df, feature_columns = prepare_dataframe(args)
    X = df[feature_columns].values.astype(np.float32)
    y = df['power_kw'].values.astype(np.float32)
    pth = df['power_theoretical'].values.astype(np.float32)

    folds = blocked_folds_indices(len(df), args.cv_folds)

    results_rows = []
    agg_metrics: Dict[str, List[Dict[str, float]]] = {}

    for fi, (tr, te) in enumerate(folds):
        fold_dir = os.path.join(outdir, f'fold_{fi+1}')
        os.makedirs(fold_dir, exist_ok=True)

        X_train = X[tr]
        y_train = y[tr]
        X_test = X[te]
        y_test = y[te]
        pth_test = pth[te]

        # reserve small val from train for catboost
        if len(X_train) > 10:
            i_val = int(len(X_train) * 0.9)
            X_tr, y_tr = X_train[:i_val], y_train[:i_val]
            X_val, y_val = X_train[i_val:], y_train[i_val:]
        else:
            X_tr, y_tr = X_train, y_train
            X_val, y_val = None, None

        # Physics-only
        y_pred_phys = pth_test
        m_phys = compute_metrics(y_test, y_pred_phys)
        results_rows.append({'fold': fi+1, 'model': 'Physics', **m_phys})
        agg_metrics.setdefault('Physics', []).append(m_phys)

        # AI-only (CatBoost direct)
        y_pred_ai, t_ai = fit_predict_catboost_direct(X_tr, y_tr, X_val, y_val, X_test, {
            'depth': args.cb_depth, 'learning_rate': args.cb_lr, 'l2_leaf_reg': args.cb_l2, 'iterations': args.cb_iter,
        })
        m_ai = compute_metrics(y_test, y_pred_ai)
        results_rows.append({'fold': fi+1, 'model': 'AI_CatBoost', **m_ai, **t_ai})
        agg_metrics.setdefault('AI_CatBoost', []).append(m_ai)

        # Hybrid (CatBoost residual + P_th)
        r_total_tr = (y_tr - pth[tr][:len(y_tr)])
        r_total_val = (y_val - pth[tr][len(y_tr):]) if X_val is not None else None
        r_total_te = (y_test - pth_test)
        y_pred_hyb, t_hyb = fit_predict_catboost_residual(X_tr, r_total_tr, X_val, r_total_val, X_test, pth_test, CatBoostConfig(
            depth=args.cb_depth, learning_rate=args.cb_lr, l2_leaf_reg=args.cb_l2, iterations=args.cb_iter
        ))
        m_hyb = compute_metrics(y_test, y_pred_hyb)
        results_rows.append({'fold': fi+1, 'model': 'Hybrid_CatBoost+Pth', **m_hyb, **t_hyb})
        agg_metrics.setdefault('Hybrid_CatBoost+Pth', []).append(m_hyb)

        # Baselines
        if args.include_baselines:
            baselines = build_baselines()
            for name, model in baselines.items():
                y_pred_b, t_b = fit_predict_sklearn(model, X_tr, y_tr, X_test)
                m_b = compute_metrics(y_test, y_pred_b)
                results_rows.append({'fold': fi+1, 'model': name, **m_b, **t_b})
                agg_metrics.setdefault(name, []).append(m_b)

    # Aggregate
    agg_summary: Dict[str, Dict[str, float]] = {}
    for model, metrics_list in agg_metrics.items():
        agg_summary[model] = {
            'MAE': float(np.mean([m['MAE'] for m in metrics_list])),
            'MSE': float(np.mean([m['MSE'] for m in metrics_list])),
            'RMSE': float(np.mean([m['RMSE'] for m in metrics_list])),
            'R2': float(np.mean([m['R2'] for m in metrics_list])),
        }

    # Save CSV and JSON
    pd.DataFrame(results_rows).to_csv(os.path.join(outdir, 'fold_metrics.csv'), index=False)
    with open(os.path.join(outdir, 'aggregate_metrics.json'), 'w') as f:
        json.dump(agg_summary, f, indent=2)

    # Plot bar chart for Physics vs AI vs Hybrid (and baselines if included)
    order = ['Physics', 'AI_CatBoost', 'Hybrid_CatBoost+Pth']
    # Append baselines in alphabetical order
    if args.include_baselines:
        extra = sorted([k for k in agg_summary.keys() if k not in order])
        order += extra
    # Reorder dict
    ordered_agg = {k: agg_summary[k] for k in order if k in agg_summary}
    plot_bar_metrics(ordered_agg, os.path.join(outdir, 'comparison_bars.png'))

    print(f"Comparison saved to: {outdir}")


if __name__ == '__main__':
    main()
