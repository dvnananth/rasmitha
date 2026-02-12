"""ReactorSim core utilities and solvers."""
from .core import (
    ReactionKinetics,
    parse_kinetics,
    simulate_batch_isothermal,
    simulate_cstr_adiabatic,
    simulate_pfr_isothermal,
)

__all__ = [
    "ReactionKinetics",
    "parse_kinetics",
    "simulate_batch_isothermal",
    "simulate_cstr_adiabatic",
    "simulate_pfr_isothermal",
]

__version__ = "0.1.0"
