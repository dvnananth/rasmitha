from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import math

Species = str


@dataclass(frozen=True)
class Arrhenius:
    A: float  # pre-exponential factor (units depend on order)
    Ea: float  # activation energy (J/mol)
    R: float = 8.314462618  # gas constant (J/mol-K)

    def k(self, T: float) -> float:
        return self.A * math.exp(-self.Ea / (self.R * T))


@dataclass(frozen=True)
class Reaction:
    """Generalized reaction with stoichiometry and optional reverse.

    stoich: mapping species -> nu (negative for reactants, positive for products)
    order: mapping species -> reaction order (power-law); if omitted, use -nu for reactants
    reverse: optional Arrhenius for reverse reaction (reversible)
    """
    forward: Arrhenius
    stoich: Dict[Species, float]
    order: Dict[Species, float] | None = None
    reverse: Arrhenius | None = None

    def rate(self, T: float, concentrations: Dict[Species, float]) -> float:
        kf = self.forward.k(T)
        # mass action / power-law for forward
        exp_forward = 1.0
        for sp, nu in self.stoich.items():
            if nu < 0:
                ord_sp = self.order[sp] if self.order and sp in self.order else -nu
                exp_forward *= max(concentrations.get(sp, 0.0), 0.0) ** ord_sp
        rf = kf * exp_forward
        if self.reverse is None:
            return rf
        # reverse term uses products
        kr = self.reverse.k(T)
        exp_rev = 1.0
        for sp, nu in self.stoich.items():
            if nu > 0:
                ord_sp = self.order[sp] if self.order and sp in self.order else nu
                exp_rev *= max(concentrations.get(sp, 0.0), 0.0) ** ord_sp
        rr = kr * exp_rev
        return rf - rr


@dataclass(frozen=True)
class Network:
    species: List[Species]
    reactions: List[Reaction]

    def rhs_batch_isothermal(self, T: float) -> Tuple[Species, ...]:
        return tuple(self.species)

    def rates(self, T: float, concentrations: Dict[Species, float]) -> List[float]:
        return [rxn.rate(T, concentrations) for rxn in self.reactions]

    def dcdt(self, T: float, conc_vector: Sequence[float]) -> List[float]:
        conc = {sp: c for sp, c in zip(self.species, conc_vector)}
        R = self.rates(T, conc)
        dcdt = [0.0 for _ in self.species]
        for r_idx, rxn in enumerate(self.reactions):
            r = R[r_idx]
            for s_idx, sp in enumerate(self.species):
                nu = rxn.stoich.get(sp, 0.0)
                if nu != 0.0:
                    dcdt[s_idx] += nu * r
        return dcdt
