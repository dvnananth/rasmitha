import argparse
import csv
from typing import List, Tuple

from .pk import OneCompIVBolus, OneCompFirstOrderAbsorption
from .pd import EmaxModel
from .engine import simulate, RegimenApplier
from .regimen import Regimen


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="PharmaSim - PK/PD simulation")
    parser.add_argument("model", choices=["iv1c", "oral1c"], help="PK model")
    parser.add_argument("--cl", type=float, required=True, help="Clearance (L/h)")
    parser.add_argument("--v", type=float, required=True, help="Volume (L)")
    parser.add_argument("--ka", type=float, default=1.0, help="Absorption rate (1/h) for oral model")
    parser.add_argument("--f", type=float, default=1.0, help="Bioavailability for oral model")
    parser.add_argument("--start", type=float, default=0.0, help="Start time (h)")
    parser.add_argument("--end", type=float, default=24.0, help="End time (h)")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step (h)")
    parser.add_argument("--dose", type=float, required=True, help="Dose amount (mg)")
    parser.add_argument("--every", type=float, default=24.0, help="Dosing interval (h)")
    parser.add_argument("--n", type=int, default=1, help="Number of doses")
    parser.add_argument("--route", choices=["iv", "oral"], default="iv", help="Dosing route")
    parser.add_argument("--infusion", type=float, default=0.0, help="Infusion duration (h) if IV infusion")
    parser.add_argument("--csv", type=str, default="output.csv", help="Output CSV path")
    parser.add_argument("--pd-emax", type=float, help="Emax for PD model")
    parser.add_argument("--pd-ec50", type=float, help="EC50 for PD model")
    parser.add_argument("--pd-baseline", type=float, default=0.0, help="Baseline effect")

    args = parser.parse_args()

    if args.model == "iv1c":
        model = OneCompIVBolus(clearance_L_per_h=args.cl, volume_L=args.v)
        state = (0.0,)  # A_c
        oral_index = None
    else:
        model = OneCompFirstOrderAbsorption(
            clearance_L_per_h=args.cl,
            volume_L=args.v,
            ka_per_h=args.ka,
            bioavailability=args.f,
        )
        state = (0.0, 0.0)  # A_gut, A_c
        oral_index = 0

    regimen = Regimen.repeated(
        start=args.start,
        every=args.every,
        n=args.n,
        amount=args.dose,
        route=args.route,
        infusion_duration=(args.infusion if args.infusion > 0 else None),
    )

    applier = RegimenApplier(regimen, state_dim=len(state), oral_state_index=oral_index)

    def rhs(t: float, s: Tuple[float, ...]) -> Tuple[float, ...]:
        return model.rhs(t, s)  # type: ignore

    result = simulate(
        rhs=rhs,
        initial_state=state,
        t0=args.start,
        t_end=args.end,
        dt=args.dt,
        dosing=applier,
        record_times=[d.time for d in regimen.doses],
    )

    # Optionally compute PD effect
    pd_vals: List[float] | None = None
    if args.pd_emax is not None and args.pd_ec50 is not None:
        pd = EmaxModel(emax=args.pd_emax, ec50=args.pd_ec50, baseline=args.pd_baseline)
        pd_vals = []
        for s in result.states:
            if args.model == "iv1c":
                conc = model.concentration((s[0],))
            else:
                conc = model.concentration((s[0], s[1]))
            pd_vals.append(pd.effect(conc))

    # Write CSV
    headers = ["time_h"]
    if args.model == "iv1c":
        headers += ["A_c_mg", "C_mg_per_L"]
    else:
        headers += ["A_gut_mg", "A_c_mg", "C_mg_per_L"]
    if pd_vals is not None:
        headers.append("effect")

    with open(args.csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for idx, (t, s) in enumerate(zip(result.times, result.states)):
            if args.model == "iv1c":
                C = model.concentration((s[0],))
                row = [t, s[0], C]
            else:
                C = model.concentration((s[0], s[1]))
                row = [t, s[0], s[1], C]
            if pd_vals is not None:
                row.append(pd_vals[idx])
            writer.writerow(row)


if __name__ == "__main__":
    run_cli()
