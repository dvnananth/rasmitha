from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

GAS_CONSTANT_R_J_PER_MOLK: float = 8.31446261815324


@dataclass
class ReactionKinetics:
    """Elementary Arrhenius kinetics for multiple reactions and species.

    Attributes
    ----------
    species: List[str]
        Species names, length n_species
    stoich: np.ndarray
        Stoichiometric coefficients matrix with shape (n_reactions, n_species)
        Each row corresponds to a reaction; negative for reactants, positive for products
    A: np.ndarray
        Pre-exponential factors for each reaction, length n_reactions
    Ea: np.ndarray
        Activation energies (J/mol) for each reaction, length n_reactions
    """

    species: List[str]
    stoich: np.ndarray
    A: np.ndarray
    Ea: np.ndarray

    @property
    def num_species(self) -> int:
        return len(self.species)

    @property
    def num_reactions(self) -> int:
        return int(self.stoich.shape[0])

    def reaction_orders(self) -> np.ndarray:
        """Reaction orders for elementary reactions: -min(nu, 0).

        Returns
        -------
        np.ndarray
            Array of shape (n_reactions, n_species) with nonnegative reaction orders
        """
        return np.maximum(-self.stoich, 0.0)

    def rate_constants(self, T: float) -> np.ndarray:
        """Arrhenius rate constants at temperature T (K)."""
        A = self.A
        Ea = self.Ea
        k = A * np.exp(-Ea / (GAS_CONSTANT_R_J_PER_MOLK * max(T, 1e-9)))
        return k

    def reaction_rates(self, c: np.ndarray, T: float) -> np.ndarray:
        """Compute reaction rates r_j = k_j * prod_i c_i^{order_ij}.

        Parameters
        ----------
        c: np.ndarray
            Species concentrations, shape (n_species,)
        T: float
            Temperature (K)
        """
        c_safe = np.clip(c, 0.0, np.inf)
        k = self.rate_constants(T)
        orders = self.reaction_orders()  # (n_rxn, n_species)
        # Compute product over species: exp(sum(orders * log(c))) with log(0)->-inf -> product 0
        # Add small epsilon for numerical stability
        eps = 1e-30
        log_c = np.log(c_safe + eps)
        exponents = np.sum(orders * log_c[None, :], axis=1)
        prod_term = np.exp(exponents)
        r = k * prod_term  # (n_rxn,)
        return r


def parse_kinetics(kin: Dict) -> ReactionKinetics:
    """Parse a session kinetics dict into ReactionKinetics.

    Expected dict keys: 'species': List[str], 'stoich': List[str], 'A': List[float], 'Ea': List[float]
    Each 'stoich' string is a comma-separated vector over species for that reaction.
    """
    species: List[str] = list(kin["species"]) if kin and "species" in kin else []
    stoich_lines: List[str] = list(kin.get("stoich", []))
    A: np.ndarray = np.asarray(kin.get("A", []), dtype=float)
    Ea: np.ndarray = np.asarray(kin.get("Ea", []), dtype=float)

    n_species = len(species)
    n_rxn = len(stoich_lines)
    if n_rxn != len(A) or n_rxn != len(Ea):
        raise ValueError("Lengths of stoich, A, and Ea must match")

    stoich_rows: List[List[float]] = []
    for line in stoich_lines:
        parts = [p.strip() for p in str(line).split(",") if p.strip() != ""]
        row = [float(x) for x in parts]
        if n_species > 0 and len(row) != n_species:
            raise ValueError(
                f"Stoich row length {len(row)} does not match number of species {n_species}"
            )
        stoich_rows.append(row)
    stoich = np.asarray(stoich_rows, dtype=float) if stoich_rows else np.zeros((0, n_species), dtype=float)

    return ReactionKinetics(species=species, stoich=stoich, A=A, Ea=Ea)


def _build_df_time_series(t: np.ndarray, c_hist: np.ndarray, T_hist: Optional[np.ndarray], species: List[str]) -> pd.DataFrame:
    df = pd.DataFrame({"t": t})
    for i, name in enumerate(species):
        df[name] = c_hist[:, i]
    if T_hist is not None:
        df["T"] = T_hist
    return df


def _build_df_space_series(z: np.ndarray, c_hist: np.ndarray, T: Optional[float], species: List[str]) -> pd.DataFrame:
    df = pd.DataFrame({"z": z})
    for i, name in enumerate(species):
        df[name] = c_hist[:, i]
    if T is not None:
        df["T"] = T
    return df


# --- Physics solvers ---

def simulate_batch_isothermal(kin: ReactionKinetics, params: Dict) -> pd.DataFrame:
    """Batch reactor, isothermal: dC/dt = nu^T r(C,T)."""
    T: float = float(params.get("T", params.get("T0", 300.0)))
    c0: np.ndarray = np.asarray(params.get("c0", [0.0] * kin.num_species), dtype=float)
    tend: float = float(params.get("tend", 10.0))
    dt: float = float(params.get("dt", 0.05))

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        c = y
        r = kin.reaction_rates(c, T)
        dc_dt = kin.stoich.T @ r
        return dc_dt

    t_eval = np.arange(0.0, max(tend, dt) + 1e-12, max(dt, 1e-6))
    sol = solve_ivp(rhs, t_span=(0.0, tend), y0=c0, t_eval=t_eval, method="RK45", vectorized=False)
    c_hist = sol.y.T
    return _build_df_time_series(sol.t, c_hist, np.full_like(sol.t, T), kin.species)


def simulate_cstr_adiabatic(kin: ReactionKinetics, params: Dict) -> pd.DataFrame:
    """CSTR dynamic adiabatic balances.

    State y = [C_1..C_n, T]
    dC/dt = (Cin - C)/tau + nu^T r(C,T)
    dT/dt = (Tin - T)/tau + (-sum_j r_j * dH_j)/rho_cp
    """
    tau: float = float(params.get("tau", 1.0))
    Tin: float = float(params.get("Tin", params.get("T0", 300.0)))
    c0: np.ndarray = np.asarray(params.get("c0", [0.0] * kin.num_species), dtype=float)
    cin: np.ndarray = np.asarray(params.get("cin", [0.0] * kin.num_species), dtype=float)
    rho_cp: float = float(params.get("rho_cp", 4000.0))
    dH: np.ndarray = np.asarray(params.get("dH", [0.0] * kin.num_reactions), dtype=float)
    tend: float = float(params.get("tend", 10.0))
    dt: float = float(params.get("dt", 0.05))

    if len(dH) != kin.num_reactions:
        raise ValueError("Length of dH must equal number of reactions")

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        c = y[:-1]
        T = float(y[-1])
        r = kin.reaction_rates(c, T)
        dc_dt = (cin - c) / max(tau, 1e-12) + kin.stoich.T @ r
        dT_dt = (Tin - T) / max(tau, 1e-12) + (-np.dot(r, dH)) / max(rho_cp, 1e-12)
        return np.concatenate([dc_dt, [dT_dt]])

    y0 = np.concatenate([c0, [Tin]])
    t_eval = np.arange(0.0, max(tend, dt) + 1e-12, max(dt, 1e-6))
    sol = solve_ivp(rhs, t_span=(0.0, tend), y0=y0, t_eval=t_eval, method="RK45", vectorized=False)
    c_hist = sol.y[:-1, :].T
    T_hist = sol.y[-1, :]
    return _build_df_time_series(sol.t, c_hist, T_hist, kin.species)


def simulate_pfr_isothermal(kin: ReactionKinetics, params: Dict) -> pd.DataFrame:
    """PFR steady profile vs residence coordinate tau (denoted z here).

    dC/dz = nu^T r(C,T), with C(0) = Cin
    """
    T: float = float(params.get("T", params.get("T0", 300.0)))
    cin: np.ndarray = np.asarray(params.get("cin", [0.0] * kin.num_species), dtype=float)
    tau_end: float = float(params.get("tau_end", params.get("z_end", 5.0)))
    dt: float = float(params.get("dt", 0.05))

    def rhs(_z: float, y: np.ndarray) -> np.ndarray:
        c = y
        r = kin.reaction_rates(c, T)
        dc_dz = kin.stoich.T @ r
        return dc_dz

    z_eval = np.arange(0.0, max(tau_end, dt) + 1e-12, max(dt, 1e-6))
    sol = solve_ivp(rhs, t_span=(0.0, tau_end), y0=cin, t_eval=z_eval, method="RK45", vectorized=False)
    c_hist = sol.y.T
    return _build_df_space_series(sol.t, c_hist, T, kin.species)
