from dataclasses import dataclass
from typing import Tuple
import math


@dataclass(frozen=True)
class OneCompIVBolus:
    """One-compartment IV bolus model.

    dA/dt = -k * A, where A is amount in central compartment (mg)
    C = A / V
    k = CL / V
    """
    clearance_L_per_h: float  # CL
    volume_L: float  # V

    def k_elim(self) -> float:
        return self.clearance_L_per_h / self.volume_L

    def rhs(self, t: float, state: Tuple[float]) -> Tuple[float]:
        A = state[0]
        dA_dt = -self.k_elim() * A
        return (dA_dt,)

    def concentration(self, state: Tuple[float]) -> float:
        return state[0] / self.volume_L


@dataclass(frozen=True)
class OneCompFirstOrderAbsorption:
    """One-compartment with first-order absorption (oral).

    States:
      A_gut: amount at absorption site (mg)
      A_c: central amount (mg)
    Equations:
      dA_gut/dt = -ka * A_gut
      dA_c/dt = F * ka * A_gut - ke * A_c
    ke = CL / V
    """
    clearance_L_per_h: float
    volume_L: float
    ka_per_h: float
    bioavailability: float = 1.0

    def k_elim(self) -> float:
        return self.clearance_L_per_h / self.volume_L

    def rhs(self, t: float, state: Tuple[float, float]) -> Tuple[float, float]:
        A_gut, A_c = state
        ka = self.ka_per_h
        ke = self.k_elim()
        dA_gut_dt = -ka * A_gut
        dA_c_dt = self.bioavailability * ka * A_gut - ke * A_c
        return (dA_gut_dt, dA_c_dt)

    def concentration(self, state: Tuple[float, float]) -> float:
        return state[1] / self.volume_L
