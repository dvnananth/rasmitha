from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class EmaxModel:
    """Simple Emax PD model with baseline.

    Effect = baseline + (Emax * C) / (EC50 + C)
    """
    emax: float
    ec50: float
    baseline: float = 0.0

    def effect(self, concentration: float) -> float:
        if concentration <= 0:
            return self.baseline
        return self.baseline + (self.emax * concentration) / (self.ec50 + concentration)
