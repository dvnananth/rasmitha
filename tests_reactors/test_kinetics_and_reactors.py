from reactorsim.kinetics import Arrhenius, Reaction, Network
from reactorsim.reactors import BatchIsothermal, CSTRIsothermal
from reactorsim.solver import integrate_ode
import numpy as np


def test_batch_simple_A_to_B():
    # A -> B with k = 1 1/h at T (choose Ea=0 so k=A)
    rxn = Reaction(forward=Arrhenius(A=1.0, Ea=0.0), stoich={"A": -1.0, "B": 1.0})
    net = Network(species=["A","B"], reactions=[rxn])
    model = BatchIsothermal(network=net, T_K=300.0)
    res = integrate_ode(model.rhs, y0=[1.0, 0.0], t_span=(0.0, 5.0), t_eval=np.linspace(0,5,51))
    A = res.y[0]
    assert A[-1] < 0.01  # mostly consumed


def test_cstr_reaches_steady_state():
    rxn = Reaction(forward=Arrhenius(A=1.0, Ea=0.0), stoich={"A": -1.0})
    net = Network(species=["A"], reactions=[rxn])
    model = CSTRIsothermal(network=net, T_K=300.0, residence_time_h=1.0, feed_conc=[1.0])
    res = integrate_ode(model.rhs, y0=[0.0], t_span=(0.0, 10.0), t_eval=np.linspace(0,10,201))
    # For first-order with tau=1, steady-state A satisfies: 0 = (A_in - A)/tau - k*A -> A = A_in / (1+k*tau) = 1/2
    A_ss = res.y[0][-1]
    assert abs(A_ss - 0.5) < 0.05
