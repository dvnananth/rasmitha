import math

from pharmasim.pk import OneCompIVBolus, OneCompFirstOrderAbsorption
from pharmasim.engine import simulate, RegimenApplier
from pharmasim.regimen import Regimen


def test_iv_bolus_exponential_decay():
    model = OneCompIVBolus(clearance_L_per_h=1.0, volume_L=10.0)
    state = (100.0,)  # 100 mg
    def rhs(t, s):
        return model.rhs(t, s)
    res = simulate(rhs, state, 0.0, 10.0, 0.1)
    k = model.k_elim()
    # Check close to analytical solution at 10 h
    expected = 100.0 * math.exp(-k * 10.0)
    assert abs(res.states[-1][0] - expected) / expected < 0.02


def test_oral_absorption_peak():
    model = OneCompFirstOrderAbsorption(clearance_L_per_h=1.0, volume_L=10.0, ka_per_h=1.5, bioavailability=0.8)
    state = (100.0, 0.0)  # 100 mg in gut
    def rhs(t, s):
        return model.rhs(t, s)
    res = simulate(rhs, state, 0.0, 24.0, 0.1)
    # Central amount should eventually increase then decrease
    amounts = [s[1] for s in res.states]
    assert max(amounts) > amounts[0]
    assert amounts[-1] < max(amounts)


def test_regimen_applier_iv_bolus():
    model = OneCompIVBolus(clearance_L_per_h=1.0, volume_L=10.0)
    state = (0.0,)
    regimen = Regimen.repeated(start=0.0, every=12.0, n=2, amount=100.0, route="iv")
    applier = RegimenApplier(regimen, state_dim=1)
    def rhs(t, s):
        return model.rhs(t, s)
    res = simulate(rhs, state, 0.0, 24.0, 0.1, dosing=applier, record_times=[0.0, 12.0])
    # Amount at t=0 should include bolus
    assert res.states[0][0] == 0.0  # before applying dose
    # First aligned step applies dose at t=0
    assert res.states[1][0] > 0.0
