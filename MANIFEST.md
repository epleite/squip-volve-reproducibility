# File manifest

- `src/sqigpi/core.py`: scale-quotient algebra used by the manuscript equations.
- `src/sqigpi/controlled.py`: compact nonlinear Volve-derived forward model.
- `tests/test_core.py`: invariance and limiting-property tests.
- `scripts/demo_quotient_geometry.py`: data-free example.
- `scripts/reproduce_controlled_geometry.py`: primary reviewer reproduction script.
- `scripts/reproduce_surrogate_sensitivity.py`: model-form sensitivity reproduction.
- `scripts/run_nonlinear_stress_test.py`: fixed-scale vs joint-nuisance nonlinear stress test.
- `data/compact/`: small ASCII Volve-derived inputs.
- `data/results/controlled/`: frozen main controlled results.
- `data/results/nonlinear/`: frozen nonlinear/posterior results.
- `data/results/surrogate/`: frozen surrogate-sensitivity results.
- `data/results/field/`: final real-fence authority table.
- `data/results/heldout/`: independent F-1A transfer-validation tables.
- `figures/`: selected supplementary figures.
- `docs/`: reproducibility and data-provenance notes.
