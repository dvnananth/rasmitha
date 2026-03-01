from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Dose:
    time: float  # hours
    amount: float  # mg
    route: str = "iv"  # "iv" (bolus) or "oral"
    infusion_duration: Optional[float] = None  # hours, for IV infusion


@dataclass(frozen=True)
class Regimen:
    doses: List[Dose]

    @staticmethod
    def repeated(start: float, every: float, n: int, amount: float, *, route: str = "iv", infusion_duration: Optional[float] = None) -> "Regimen":
        """Create a repeated dosing regimen.
        - start: first dose time (h)
        - every: interval (h)
        - n: number of doses
        - amount: dose amount (mg)
        - route: "iv" or "oral"
        - infusion_duration: hours if IV infusion
        """
        doses = []
        for i in range(n):
            doses.append(Dose(time=start + i * every, amount=amount, route=route, infusion_duration=infusion_duration))
        return Regimen(doses=doses)

    def next_dose_index(self, t: float, start_idx: int = 0) -> int:
        for i in range(start_idx, len(self.doses)):
            if self.doses[i].time >= t:
                return i
        return len(self.doses)
