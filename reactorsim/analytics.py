from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple, List
import random
import time

import numpy as np

from .kinetics import Network
from .reactors import CSTRIsothermal
from .solver import integrate_ode


@dataclass(frozen=True)
class Dataset:
    feature_names: List[str]
    X: np.ndarray  # (n_samples, n_features)
    y_true: np.ndarray  # (n_samples,)
    y_low: np.ndarray  # (n_samples,)
    low_pred_time_s: float
    true_time_s: float


def generate_cstr_isothermal_dataset(
    network: Network,
    species_index: int,
    T_range: Tuple[float, float],
    tau_range: Tuple[float, float],
    feed_ranges: Sequence[Tuple[float, float]],
    *,
    n_samples: int = 300,
    t_end_h: float = 10.0,
    dt_high: float = 0.01,
    dt_low: float = 0.1,
    seed: int | None = 42,
) -> Dataset:
    """Generate a dataset by simulating a CSTR (isothermal) over a grid of inputs.

    Features: [T_K, tau_h, c_in_species...]
    Targets: final concentration of species_index at t_end.
    y_true uses fine dt_high (high fidelity). y_low uses coarser dt_low (baseline physics).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n_species = len(network.species)
    if len(feed_ranges) != n_species:
        raise ValueError("feed_ranges must have one (min, max) tuple per species")
    if not (0 <= species_index < n_species):
        raise ValueError("species_index out of range")

    feature_names = ["T_K", "tau_h"] + [f"c_in_{sp}" for sp in network.species]
    X = np.zeros((n_samples, 2 + n_species), dtype=float)
    y_true = np.zeros((n_samples,), dtype=float)
    y_low = np.zeros((n_samples,), dtype=float)

    c0 = [0.0 for _ in range(n_species)]

    # Low fidelity
    low_start = time.perf_counter()
    for i in range(n_samples):
        T = random.uniform(*T_range)
        tau = random.uniform(*tau_range)
        c_in = [random.uniform(*rng) for rng in feed_ranges]
        X[i, 0] = T
        X[i, 1] = tau
        X[i, 2:] = c_in
        model = CSTRIsothermal(network=network, T_K=T, residence_time_h=tau, feed_conc=c_in)
        t_eval_low = np.arange(0.0, t_end_h + 1e-12, dt_low)
        res_low = integrate_ode(model.rhs, y0=c0, t_span=(0.0, t_end_h), t_eval=t_eval_low)
        y_low[i] = res_low.y[species_index][-1]
    low_time = time.perf_counter() - low_start

    # High fidelity
    high_start = time.perf_counter()
    for i in range(n_samples):
        T = X[i, 0]
        tau = X[i, 1]
        c_in = X[i, 2:].tolist()
        model = CSTRIsothermal(network=network, T_K=T, residence_time_h=tau, feed_conc=c_in)
        t_eval_high = np.arange(0.0, t_end_h + 1e-12, dt_high)
        res_high = integrate_ode(model.rhs, y0=c0, t_span=(0.0, t_end_h), t_eval=t_eval_high)
        y_true[i] = res_high.y[species_index][-1]
    high_time = time.perf_counter() - high_start

    return Dataset(
        feature_names=feature_names,
        X=X,
        y_true=y_true,
        y_low=y_low,
        low_pred_time_s=low_time,
        true_time_s=high_time,
    )
