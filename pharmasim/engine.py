from typing import Callable, List, Sequence, Tuple, Optional
from dataclasses import dataclass

from .regimen import Regimen


State = Tuple[float, ...]
RHSFunction = Callable[[float, State], State]
DosingApplier = Callable[[float, State], State]


@dataclass(frozen=True)
class SimulationResult:
    times: List[float]
    states: List[State]


def rk4_step(rhs: RHSFunction, t: float, state: State, h: float) -> State:
    k1 = rhs(t, state)
    s2 = tuple(s + h * k1_i / 2.0 for s, k1_i in zip(state, k1))
    k2 = rhs(t + h / 2.0, s2)
    s3 = tuple(s + h * k2_i / 2.0 for s, k2_i in zip(state, k2))
    k3 = rhs(t + h / 2.0, s3)
    s4 = tuple(s + h * k3_i for s, k3_i in zip(state, k3))
    k4 = rhs(t + h, s4)
    next_state = tuple(
        s + (h / 6.0) * (k1_i + 2.0 * k2_i + 2.0 * k3_i + k4_i)
        for s, k1_i, k2_i, k3_i, k4_i in zip(state, k1, k2, k3, k4)
    )
    return next_state


def simulate(
    rhs: RHSFunction,
    initial_state: State,
    t0: float,
    t_end: float,
    dt: float,
    dosing: Optional[Callable[[float, State], State]] = None,
    record_times: Optional[Sequence[float]] = None,
) -> SimulationResult:
    """Simulate ODE system with optional dosing events using RK4 fixed step.

    - rhs: derivative function
    - initial_state: tuple of state values
    - t0, t_end: start and end time (hours)
    - dt: fixed time step (hours)
    - dosing: function to apply instantaneous dose at time t
    - record_times: optional additional times to record exactly
    """
    times: List[float] = []
    states: List[State] = []

    t = t0
    state = initial_state

    def maybe_record(curr_t: float, curr_state: State) -> None:
        times.append(curr_t)
        states.append(curr_state)

    # Create a set of event times to align step boundaries
    event_times = set([t0, t_end])
    if record_times is not None:
        event_times.update(record_times)
    # Always include dosing times if dosing is a regimen applier
    # The dosing function can expose an attribute with times for alignment
    dose_times = getattr(dosing, "times", None)
    if dose_times is not None:
        event_times.update(dose_times)

    # Generate aligned time grid
    grid = sorted(event_times)
    # Fill in between with uniform dt if needed
    full_grid: List[float] = []
    for i in range(len(grid) - 1):
        a = grid[i]
        b = grid[i + 1]
        full_grid.append(a)
        n = max(1, int(round((b - a) / dt)))
        h = (b - a) / n
        for k in range(1, n):
            full_grid.append(a + k * h)
    full_grid.append(grid[-1])

    # Main loop across full grid
    maybe_record(t, state)
    for i in range(len(full_grid) - 1):
        a = full_grid[i]
        b = full_grid[i + 1]
        h = b - a
        # Apply dosing at the beginning of the interval if exact match
        if dosing is not None and hasattr(dosing, "apply_if_due"):
            state = dosing.apply_if_due(a, state)
        state = rk4_step(rhs, a, state, h)
        t = b
        maybe_record(t, state)

    return SimulationResult(times=times, states=states)


class RegimenApplier:
    """Callable dosing applier to use with simulate(). Handles bolus and infusions.
    Exposes .times for alignment and .apply_if_due(t, state) to apply doses.
    """

    def __init__(self, regimen: Regimen, state_dim: int, oral_state_index: Optional[int] = None):
        self.regimen = regimen
        self.state_dim = state_dim
        self.oral_state_index = oral_state_index  # index of gut amount for oral dosing
        self.times = [d.time for d in regimen.doses]

    def __call__(self, t: float, state: State) -> State:
        # not used by engine directly; engine uses apply_if_due
        return state

    def apply_if_due(self, t: float, state: State) -> State:
        # apply any dose exactly at time t
        new_state = list(state)
        for dose in self.regimen.doses:
            if abs(dose.time - t) < 1e-9:
                if dose.route == "iv":
                    if dose.infusion_duration is None or dose.infusion_duration == 0:
                        # bolus to central compartment (assume index 0 or 1 depending on model)
                        target_index = 0 if self.oral_state_index is None else 1
                        new_state[target_index] += dose.amount
                    else:
                        # For simplicity, split infusion into small boluses across the interval.
                        # The engine aligns time grid at infusion start and end, but we subdivide here by dt-like chunks.
                        # Since engine doesn't pass dt, approximate over 20 substeps.
                        substeps = 20
                        increment = dose.amount / substeps
                        dt = dose.infusion_duration / substeps
                        time_accum = dose.time
                        for _ in range(substeps):
                            if abs(time_accum - t) < 1e-9:
                                target_index = 0 if self.oral_state_index is None else 1
                                new_state[target_index] += increment
                            time_accum += dt
                elif dose.route == "oral":
                    if self.oral_state_index is None:
                        raise ValueError("oral_state_index must be provided for oral dosing models")
                    new_state[self.oral_state_index] += dose.amount
                else:
                    raise ValueError(f"Unsupported route: {dose.route}")
        return tuple(new_state)
