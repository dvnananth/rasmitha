from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

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
