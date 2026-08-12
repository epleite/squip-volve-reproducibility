# Reproducibility map

## Level 1 — core method, no external data

```bash
python -m pip install -e ".[test]"
pytest -q
python scripts/demo_quotient_geometry.py
```

This verifies the horizontal-lift/Schur-complement energy identity, nuisance-coordinate invariance, positive-semidefinite information loss, and the eigendirectional authority formula.

## Level 2 — controlled Volve-derived benchmark

```bash
python scripts/reproduce_controlled_geometry.py
python scripts/reproduce_surrogate_sensitivity.py
```

The scripts use only the ASCII tables in `data/compact/`. They refit the reference petroelastic surrogate and reconstruct the controlled forward response and local quotient geometry. They then compare regenerated values with the frozen paper-level tables and fail loudly if the agreement is outside numerical tolerance.

The slower nonlinear test is available as

```bash
python scripts/run_nonlinear_stress_test.py --nmc 5   # smoke test
python scripts/run_nonlinear_stress_test.py --nmc 50  # paper-sized ensemble
```

The exact paper-level outputs from the final 50-realization run and posterior sampling are frozen under `data/results/nonlinear/` so reviewers can inspect the reported results without repeating the expensive calculation.

## Level 3 — field-scale products

Raw Volve prestack seismic data are not mirrored. Obtain them from the official Equinor Volve release and follow `docs/VOLVE_DATA.md`. Final numerical field-authority and F-1A validation tables are included under `data/results/field/` and `data/results/heldout/`.

## Determinism

All stochastic scripts expose or fix their random seeds. The paper-level nonlinear robustness workflow uses seed `20260812`.
