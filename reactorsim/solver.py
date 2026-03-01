from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class SolveResult:
    t: List[float]
    y: List[List[float]]  # y[i][k] species i at time k
    status: int
    message: str


def integrate_ode(
    rhs: Callable[[float, Sequence[float]], Sequence[float]],
    y0: Sequence[float],
    t_span: Tuple[float, float],
    t_eval: Optional[Sequence[float]] = None,
    method: str = "LSODA",
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> SolveResult:
    if method.upper() == "LSODA":
        method_name = "LSODA"
    elif method.upper() == "BDF":
        method_name = "BDF"
    else:
        method_name = method
    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y),
        y0=np.asarray(y0, dtype=float),
        t_span=t_span,
        t_eval=np.asarray(t_eval, dtype=float) if t_eval is not None else None,
        method=method_name,
        rtol=rtol,
        atol=atol,
    )
    return SolveResult(t=sol.t.tolist(), y=sol.y.tolist(), status=sol.status, message=sol.message)
