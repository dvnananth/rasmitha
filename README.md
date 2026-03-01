# PharmaSim

A lightweight research-oriented PK/PD simulation toolkit with a simple CLI.

## Features
- One-compartment IV bolus PK model
- One-compartment oral with first-order absorption
- Dosing regimens (bolus and infusion [approx])
- Fixed-step RK4 solver with event alignment
- Optional PD Emax linkage
- CSV export via CLI

## Quickstart

Install dependencies:

```bash
pip install -r requirements.txt
```

Run an IV bolus simulation (2 doses of 100 mg every 12 h, CL=1 L/h, V=10 L):

```bash
python -m pharmasim.cli iv1c --cl 1 --v 10 --dose 100 --n 2 --every 12 --csv out.csv
```

Or an oral model with ka and F:

```bash
python -m pharmasim.cli oral1c --cl 1 --v 10 --ka 1.5 --f 0.8 --dose 100 --route oral --n 1 --csv out.csv
```

Add a PD effect curve:

```bash
python -m pharmasim.cli iv1c --cl 1 --v 10 --dose 100 --n 1 --pd-emax 1.0 --pd-ec50 2.0 --csv out.csv
```

## Testing

```bash
pytest -q
```
