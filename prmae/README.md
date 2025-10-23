# PR-MAE: Physics-Residual Multi-scale Adaptive Ensemble

This repository implements the PR-MAE architecture for wind turbine power prediction with physics-aware multi-scale residual decomposition and adaptive attention fusion.

## Structure

- `prmae/data`: data loaders, preprocessing, power curve utilities
- `prmae/models`: PINN residual learners, attention fusion, CatBoost wrapper
- `prmae/losses`: physics, boundary, scale-consistency, attention regularizers
- `prmae/training`: training loops for pretraining and joint training
- `prmae/evaluation`: metrics, ablations
- `prmae/cli`: command-line entry points

## Quickstart

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare data: place CSVs as described or pass paths via CLI.

3. Train:
```bash
python -m prmae.cli.train --data.t1 "/path/to/T1.csv" --data.power_curve "/path/to/GE Turbine Power Curve.csv" --data.texas "/path/to/TexasTurbine.csv"
```

4. Evaluate:
```bash
python -m prmae.cli.evaluate --ckpt /path/to/checkpoint.ckpt
```
