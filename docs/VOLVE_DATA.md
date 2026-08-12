# Volve data provenance

The field application uses the public Equinor Volve release, with prepared 15/9-19 A and 15/9-19 BT2 fences following the public Seis2Rock Volve workflow. The independent transfer test uses public 15/9-F-1 A well logs and formation picks.

## Official source

Use the current **Volve field data set** page on the Equinor website and follow Equinor's access instructions and Open Data Licence. The raw Volve release is not mirrored in this repository.

## Compact reviewer benchmark

`data/compact/` contains only the small ASCII well/log windows needed to reconstruct the controlled experiment:

- `19A_training_window.csv` — 19A samples within Hugin +/- 120 m used to fit the reference surrogate.
- `BT2_training_window.csv` — BT2 samples within Hugin +/- 120 m used in the same fit.
- `19A_controlled_window.csv` — 54-sample 19A controlled window, with an `in_hugin` indicator.
- `benchmark_metadata.json` — fixed prior scales, nuisance prior, noise correlation, wavelet and scenario metadata.

The compact tables contain depth, porosity, shale volume, water saturation, P- and S-wave velocities, and density. They exist solely to make the controlled paper benchmark reviewer-runnable. Their provenance remains the Equinor Volve release; see `DATA_LICENSE.md`.

## Field reproduction

The full field figures additionally require the prestack fence arrays, horizons, and local well gathers. Because these originate from the much larger public release, the repository freezes the final field numerical outputs in `data/results/field/` and documents the compact scientific benchmark separately.
