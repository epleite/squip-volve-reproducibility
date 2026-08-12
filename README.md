# SQuIP: Scale-Quotient Information-Geometric Petrophysical Inversion — Volve Reproducibility Package

This repository accompanies the manuscript **“Scale-Quotient Information-Geometric Petrophysical Inversion: Separating Petrophysical Information from Seismic Scale Uncertainty.”** It is designed so a reviewer can verify the mathematical construction and rerun the compact Volve-derived benchmark without downloading the full Volve archive.

## What this repository reproduces

The code implements the nuisance-profiled information metric

\[
G_Q = G_{uu} - G_{u\eta}G_{\eta\eta}^{-1}G_{\eta u},
\]

its horizontal-lift interpretation, eigendirectional authority \(A_i^Q=\lambda_i^Q/(1+\lambda_i^Q)\), and the property-axis authority matrix \(A_Q=G_Q(I+G_Q)^{-1}\).

Three layers of verification are provided:

1. **Core/no-data tests** — Schur-complement identities, invariance to nuisance reparameterization, positive-semidefinite information loss, and authority formula.
2. **Compact Volve-derived controlled benchmark** — polynomial petroelastic surrogate, Gaussian-weighted Backus VTI homogenization, Rüger PP response, 22-Hz wavelet, correlated noise, and quotient geometry at 4, 5, and 7 m.
3. **Archived paper-level outputs** — nonlinear stress-test, posterior-sampling, surrogate-sensitivity, held-out F-1A, and final field-authority tables.

## Quick start

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
pytest -q
python scripts/demo_quotient_geometry.py
python scripts/reproduce_controlled_geometry.py
python scripts/reproduce_surrogate_sensitivity.py
```

A successful controlled run ends with:

```text
PASS: compact ASCII benchmark reproduces archived local geometry.
```

The nonlinear inversion is deliberately separate because it is slower:

```bash
# quick reviewer smoke test
python scripts/run_nonlinear_stress_test.py --nmc 5

# paper-sized nonlinear ensemble
python scripts/run_nonlinear_stress_test.py --nmc 50
```

## Expected controlled geometry

The portable nonlinear check reproduces the archived local geometry used in the paper's nonlinear robustness test. Representative values are:

| true scale | A1_Q | A2_Q | A3_Q |
|---:|---:|---:|---:|
| 4 m | ~0.9990 | ~0.9975 | ~0.713 |
| 5 m | ~0.9987 | ~0.9967 | ~0.662 |
| 7 m | ~0.9978 | ~0.9939 | ~0.521 |

The larger 5000-realization linear-Gaussian experiment and its exact paper values are preserved in `data/results/controlled/`.

## Repository map

```text
src/sqigpi/core.py                 central quotient-information algebra
src/sqigpi/controlled.py           compact Volve-derived nonlinear forward
scripts/demo_quotient_geometry.py  no-data mathematical example
scripts/reproduce_controlled_geometry.py
scripts/run_nonlinear_stress_test.py
scripts/reproduce_surrogate_sensitivity.py
tests/test_core.py
data/compact/                       reviewer-runnable ASCII inputs
data/results/                       frozen numerical outputs used in the paper
figures/                            selected supplementary robustness figures
docs/                               data provenance and reproduction notes
```

## Data provenance and licensing

The software is MIT licensed. **No Volve-derived data are relicensed under MIT.** The compact ASCII benchmark inputs remain subject to the current Equinor Open Data Licence and attribution/use conditions; see `DATA_LICENSE.md`, `data/README.md`, and `docs/VOLVE_DATA.md`. Raw Volve seismic volumes are **not** redistributed.

## Reproducibility philosophy

The repository contains only the scientific implementation, compact benchmark inputs, frozen outputs, tests, and documentation needed to understand and verify the paper. Manuscript source, internal research logs, temporary notebooks, and development artifacts are intentionally excluded.

## Citation

See `CITATION.cff`. Please cite the associated GEOPHYSICS paper once bibliographic details are assigned.
