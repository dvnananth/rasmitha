from reactorsim.kinetics import Arrhenius, Reaction, Network
from reactorsim.reactors import PFRIsothermal, BatchAdiabatic, CSTRAdiabatic
from reactorsim.solver import integrate_ode
import numpy as np


def test_pfr_converts_A_to_B_with_residence_time():
    rxn = Reaction(forward=Arrhenius(A=1.0, Ea=0.0), stoich={"A": -1.0, "B": 1.0})
    net = Network(species=["A", "B"], reactions=[rxn])
    model = PFRIsothermal(network=net, T_K=300.0)
    res = integrate_ode(model.rhs, y0=[1.0, 0.0], t_span=(0.0, 3.0), t_eval=np.linspace(0, 3, 61))
    # Expect substantial conversion at tau=3 for k=1
    assert res.y[0][-1] < 0.1


def test_batch_adiabatic_temperature_rises_for_exothermic():
    rxn = Reaction(forward=Arrhenius(A=1.0, Ea=0.0), stoich={"A": -1.0})
    net = Network(species=["A"], reactions=[rxn])
    model = BatchAdiabatic(network=net, T0_K=300.0, rho_cp=4000.0, dH_rxn_J_per_mol=[-50000.0])
    res = integrate_ode(model.rhs, y0=[1.0, 300.0], t_span=(0.0, 2.0), t_eval=np.linspace(0, 2, 101))
    assert res.y[1][-1] > 300.0


def test_cstr_adiabatic_temperature_coupling():
    rxn = Reaction(forward=Arrhenius(A=1.0, Ea=0.0), stoich={"A": -1.0})
    net = Network(species=["A"], reactions=[rxn])
    model = CSTRAdiabatic(network=net, T_in_K=300.0, residence_time_h=1.0, feed_conc=[1.0], rho_cp=4000.0, dH_rxn_J_per_mol=[-50000.0])
    res = integrate_ode(model.rhs, y0=[0.0, 300.0], t_span=(0.0, 5.0), t_eval=np.linspace(0,5,101))
    # Temperature should deviate from inlet due to reaction heat
    assert abs(res.y[1][-1] - 300.0) > 0.1
