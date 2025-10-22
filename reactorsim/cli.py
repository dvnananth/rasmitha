import argparse
import csv
from typing import List

import numpy as np

from .kinetics import Arrhenius, Reaction, Network
from .reactors import BatchIsothermal, CSTRIsothermal, PFRIsothermal, BatchAdiabatic, CSTRAdiabatic
from .solver import integrate_ode
from .data import load_stirred_reactor_cfd_csv


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="ReactorSim - reactor modeling CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Batch
    p_batch = sub.add_parser("batch", help="Batch isothermal reactor")
    p_batch.add_argument("--species", nargs='+', required=True)
    p_batch.add_argument("--stoich", nargs='+', help="Reactions as comma-separated nu list per species order; e.g., '-1,1'", required=True)
    p_batch.add_argument("--A", nargs='+', type=float, required=True)
    p_batch.add_argument("--Ea", nargs='+', type=float, required=True)
    p_batch.add_argument("--T", type=float, required=True)
    p_batch.add_argument("--c0", nargs='+', type=float, required=True)
    p_batch.add_argument("--tend", type=float, default=1.0)
    p_batch.add_argument("--dt", type=float, default=0.01)
    p_batch.add_argument("--csv", type=str, default="batch.csv")

    # CSTR
    p_cstr = sub.add_parser("cstr", help="CSTR isothermal reactor")
    p_cstr.add_argument("--species", nargs='+', required=True)
    p_cstr.add_argument("--stoich", nargs='+', required=True)
    p_cstr.add_argument("--A", nargs='+', type=float, required=True)
    p_cstr.add_argument("--Ea", nargs='+', type=float, required=True)
    p_cstr.add_argument("--T", type=float, required=True)
    p_cstr.add_argument("--c0", nargs='+', type=float, required=True)
    p_cstr.add_argument("--cin", nargs='+', type=float, required=True)
    p_cstr.add_argument("--tau", type=float, required=True)
    p_cstr.add_argument("--tend", type=float, default=10.0)
    p_cstr.add_argument("--dt", type=float, default=0.05)
    p_cstr.add_argument("--csv", type=str, default="cstr.csv")

    # PFR (isothermal) over residence time
    p_pfr = sub.add_parser("pfr", help="PFR isothermal (residence time coordinate)")
    p_pfr.add_argument("--species", nargs='+', required=True)
    p_pfr.add_argument("--stoich", nargs='+', required=True)
    p_pfr.add_argument("--A", nargs='+', type=float, required=True)
    p_pfr.add_argument("--Ea", nargs='+', type=float, required=True)
    p_pfr.add_argument("--T", type=float, required=True)
    p_pfr.add_argument("--cin", nargs='+', type=float, required=True)
    p_pfr.add_argument("--tau_end", type=float, required=True)
    p_pfr.add_argument("--dt", type=float, default=0.01)
    p_pfr.add_argument("--csv", type=str, default="pfr.csv")

    # Batch adiabatic
    p_badia = sub.add_parser("batch-adiabatic", help="Batch adiabatic reactor")
    p_badia.add_argument("--species", nargs='+', required=True)
    p_badia.add_argument("--stoich", nargs='+', required=True)
    p_badia.add_argument("--A", nargs='+', type=float, required=True)
    p_badia.add_argument("--Ea", nargs='+', type=float, required=True)
    p_badia.add_argument("--T0", type=float, required=True)
    p_badia.add_argument("--c0", nargs='+', type=float, required=True)
    p_badia.add_argument("--rho_cp", type=float, required=True)
    p_badia.add_argument("--dH", nargs='+', type=float, required=True)
    p_badia.add_argument("--tend", type=float, default=1.0)
    p_badia.add_argument("--dt", type=float, default=0.01)
    p_badia.add_argument("--csv", type=str, default="batch_adiabatic.csv")

    # CSTR adiabatic
    p_cadia = sub.add_parser("cstr-adiabatic", help="CSTR adiabatic reactor")
    p_cadia.add_argument("--species", nargs='+', required=True)
    p_cadia.add_argument("--stoich", nargs='+', required=True)
    p_cadia.add_argument("--A", nargs='+', type=float, required=True)
    p_cadia.add_argument("--Ea", nargs='+', type=float, required=True)
    p_cadia.add_argument("--Tin", type=float, required=True)
    p_cadia.add_argument("--c0", nargs='+', type=float, required=True)
    p_cadia.add_argument("--cin", nargs='+', type=float, required=True)
    p_cadia.add_argument("--tau", type=float, required=True)
    p_cadia.add_argument("--rho_cp", type=float, required=True)
    p_cadia.add_argument("--dH", nargs='+', type=float, required=True)
    p_cadia.add_argument("--tend", type=float, default=10.0)
    p_cadia.add_argument("--dt", type=float, default=0.05)
    p_cadia.add_argument("--csv", type=str, default="cstr_adiabatic.csv")

    # Data loader
    p_data = sub.add_parser("load-cfd", help="Load stirred reactor CFD dataset and print summary")
    p_data.add_argument("--csv", type=str, required=True)

    args = parser.parse_args()

    if args.cmd == "load-cfd":
        rows = load_stirred_reactor_cfd_csv(args.csv)
        print(f"Loaded {len(rows)} rows. RPM range: {min(r.rpm for r in rows)} - {max(r.rpm for r in rows)}")
        return

    # Build network
    species = args.species
    # Parse stoich: each reaction string is comma-separated nu per species order
    reactions: List[Reaction] = []
    if len(args.stoich) != len(args.A) or len(args.stoich) != len(args.Ea):
        raise SystemExit("stoich, A, Ea must have equal counts")
    for r_str, A, Ea in zip(args.stoich, args.A, args.Ea):
        nums = [float(x) for x in r_str.split(',')]
        if len(nums) != len(species):
            raise SystemExit("Each stoich vector must match number of species")
        sto = {sp: nu for sp, nu in zip(species, nums)}
        reactions.append(Reaction(forward=Arrhenius(A=A, Ea=Ea), stoich=sto))
    net = Network(species=species, reactions=reactions)

    if args.cmd == "batch":
        model = BatchIsothermal(network=net, T_K=args.T)
        t_eval = np.arange(0.0, args.tend + 1e-12, args.dt)
        res = integrate_ode(model.rhs, y0=args.c0, t_span=(0.0, args.tend), t_eval=t_eval)
        with open(args.csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["time"] + species)
            for k, t in enumerate(res.t):
                row = [t] + [res.y[i][k] for i in range(len(species))]
                writer.writerow(row)
        return

    if args.cmd == "cstr":
        model = CSTRIsothermal(network=net, T_K=args.T, residence_time_h=args.tau, feed_conc=args.cin)
        t_eval = np.arange(0.0, args.tend + 1e-12, args.dt)
        res = integrate_ode(model.rhs, y0=args.c0, t_span=(0.0, args.tend), t_eval=t_eval)
        with open(args.csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["time"] + species)
            for k, t in enumerate(res.t):
                row = [t] + [res.y[i][k] for i in range(len(species))]
                writer.writerow(row)
        return

    if args.cmd == "pfr":
        model = PFRIsothermal(network=net, T_K=args.T)
        t_eval = np.arange(0.0, args.tau_end + 1e-12, args.dt)
        res = integrate_ode(model.rhs, y0=args.cin, t_span=(0.0, args.tau_end), t_eval=t_eval)
        with open(args.csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["tau"] + species)
            for k, t in enumerate(res.t):
                row = [t] + [res.y[i][k] for i in range(len(species))]
                writer.writerow(row)
        return

    if args.cmd == "batch-adiabatic":
        model = BatchAdiabatic(network=net, T0_K=args.T0, rho_cp=args.rho_cp, dH_rxn_J_per_mol=args.dH)
        y0 = list(args.c0) + [args.T0]
        t_eval = np.arange(0.0, args.tend + 1e-12, args.dt)
        res = integrate_ode(model.rhs, y0=y0, t_span=(0.0, args.tend), t_eval=t_eval)
        with open(args.csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["time"] + species + ["T_K"])
            for k, t in enumerate(res.t):
                row = [t] + [res.y[i][k] for i in range(len(species))] + [res.y[len(species)][k]]
                writer.writerow(row)
        return

    if args.cmd == "cstr-adiabatic":
        model = CSTRAdiabatic(
            network=net,
            T_in_K=args.Tin,
            residence_time_h=args.tau,
            feed_conc=args.cin,
            rho_cp=args.rho_cp,
            dH_rxn_J_per_mol=args.dH,
        )
        y0 = list(args.c0) + [args.Tin]
        t_eval = np.arange(0.0, args.tend + 1e-12, args.dt)
        res = integrate_ode(model.rhs, y0=y0, t_span=(0.0, args.tend), t_eval=t_eval)
        with open(args.csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["time"] + species + ["T_K"])
            for k, t in enumerate(res.t):
                row = [t] + [res.y[i][k] for i in range(len(species))] + [res.y[len(species)][k]]
                writer.writerow(row)
        return


if __name__ == "__main__":
    run_cli()
