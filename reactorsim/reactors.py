from dataclasses import dataclass
from typing import List, Sequence

from .kinetics import Network

Vector = List[float]


@dataclass(frozen=True)
class BatchIsothermal:
    network: Network
    T_K: float

    def rhs(self, t: float, c: Sequence[float]) -> Vector:
        return self.network.dcdt(self.T_K, c)


@dataclass(frozen=True)
class CSTRIsothermal:
    network: Network
    T_K: float
    residence_time_h: float
    feed_conc: Sequence[float]

    def rhs(self, t: float, c: Sequence[float]) -> Vector:
        r = self.network.dcdt(self.T_K, c)
        # CSTR: dc/dt = (c_in - c)/tau + sum(nu_i * r)
        return [ (c_in - c_i)/self.residence_time_h + r_i for c_i, c_in, r_i in zip(c, self.feed_conc, r) ]


@dataclass(frozen=True)
class PFRIsothermal:
    """Plug Flow Reactor expressed over residence time coordinate tau (h).

    dc/dtau = r(c, T)
    """
    network: Network
    T_K: float

    def rhs(self, tau: float, c: Sequence[float]) -> Vector:
        return self.network.dcdt(self.T_K, c)


@dataclass(frozen=True)
class BatchAdiabatic:
    """Batch reactor with adiabatic energy balance.

    dT/dt = -(1/(rho*Cp)) * sum(dH_j * r_j)
    where dH_j is reaction enthalpy (J/mol), r_j is rate (mol/(L*h))
    Units assumed consistent; rho*Cp should match concentration units scaling.
    """
    network: Network
    T0_K: float
    rho_cp: float  # product of density and heat capacity (J/L-K)
    dH_rxn_J_per_mol: Sequence[float]

    def rhs(self, t: float, y: Sequence[float]) -> Vector:
        # state y = [c_0..c_{N-1}, T]
        n = len(self.network.species)
        c = list(y[:n])
        T = y[n]
        dcdt = self.network.dcdt(T, c)
        rates = self.network.rates(T, {sp: ci for sp, ci in zip(self.network.species, c)})
        heat_source = 0.0
        for dh, r in zip(self.dH_rxn_J_per_mol, rates):
            heat_source += dh * r
        dTdt = -(1.0 / self.rho_cp) * heat_source
        return list(dcdt) + [dTdt]


@dataclass(frozen=True)
class CSTRAdiabatic:
    """CSTR with adiabatic energy balance.

    dc/dt = (c_in - c)/tau + nu*r
    dT/dt = (T_in - T)/tau - (1/(rho*Cp)) * sum(dH_j * r_j)
    """
    network: Network
    T_in_K: float
    residence_time_h: float
    feed_conc: Sequence[float]
    rho_cp: float  # J/L-K
    dH_rxn_J_per_mol: Sequence[float]

    def rhs(self, t: float, y: Sequence[float]) -> Vector:
        n = len(self.network.species)
        c = list(y[:n])
        T = y[n]
        # species
        r_vec = self.network.dcdt(T, c)
        dcdt = [ (c_in - c_i)/self.residence_time_h + r_i for c_i, c_in, r_i in zip(c, self.feed_conc, r_vec) ]
        # energy
        rates = self.network.rates(T, {sp: ci for sp, ci in zip(self.network.species, c)})
        heat_source = 0.0
        for dh, r in zip(self.dH_rxn_J_per_mol, rates):
            heat_source += dh * r
        dTdt = (self.T_in_K - T)/self.residence_time_h - (1.0 / self.rho_cp) * heat_source
        return list(dcdt) + [dTdt]
