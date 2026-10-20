# PCA Coefficient Optimization for Photonic Design

This repository contains scripts for PCA-based optimization in two settings:

1. A standalone analytical example.
2. A photonic inverse-design workflow coupled to Lumerical FDTD.

The photonic workflow represents device geometries using a reduced number of PCA coefficients. These coefficients are reconstructed into full device-index maps and evaluated using Lumerical FDTD simulations.

## Repository status

The repository currently uses runnable scripts in the `scripts/` directory.

The `src/` directory is reserved for future refactoring into reusable Python modules. At present, no active scripts have been moved into `src/`.

Temporary Lumerical `.fsp` files generated during optimization are not included in the repository. The `temp_sessions` files were removed before publishing to keep the repository lightweight. These files are generated locally when the optimization script is run.

## Repository structure

```text
pca-coeff-opt/
├── data/
│   ├── models/
│   │   ├── cumulative_explained_variance.npy
│   │   ├── explained_variance_ratio.npy
│   │   ├── photonic_coefficients.npy
│   │   ├── photonic_pca.joblib
│   │   └── reconstructed_ensemble.npy
│   ├── processed/
│   └── raw/
│       ├── ensemble_data.npy
│       └── fsp/
│
├── docs/
│
├── lumerical/
│   ├── bend_variable_base.lsf
│   ├── optimization_base_norun.lsf
│   └── optimization_base_run.lsf
│
├── results/
│   ├── analytical/
│   └── photonics/
│       ├── optimization/
│       ├── pca/
│       └── dataset_extremes.png
│
├── scripts/
│   ├── fit_photonic_pca.py
│   ├── generate_photonic_devices.py
│   ├── optimize_photonic_device.py
│   └── run_analytical_example.py
│
└── src/
    └── Reserved for future reusable modules
```

## Directory descriptions

### `data/models/`

Contains the saved PCA model and PCA-related data used by the photonic workflow.

Important files include:

- `photonic_pca.joblib`: fitted PCA model.
- `photonic_coefficients.npy`: PCA coefficients for the photonic device ensemble.
- `reconstructed_ensemble.npy`: reconstructed device ensemble.
- `explained_variance_ratio.npy`: variance explained by each PCA component.
- `cumulative_explained_variance.npy`: cumulative explained variance.

### `data/raw/`

Contains raw input data used to generate or analyze the photonic device ensemble.

The `fsp/` directory is reserved for raw Lumerical simulation files when they are required locally. Large simulation files should generally not be committed unless they are intentionally included for reproducibility.

### `lumerical/`

Contains the Lumerical script templates used by the Python optimization workflow.

- `optimization_base_run.lsf`: initializes and runs the base simulation. It is used to retrieve the monitor dimensions and verify that the Lumerical monitor grid matches the PCA matrix dimensions.
- `optimization_base_norun.lsf`: initializes the simulation without running it. It is recalled during the optimization iterations so that Python can inject a new PCA-reconstructed geometry and run the forward and adjoint simulations.
- `bend_variable_base.lsf`: base Lumerical script for the variable photonic bend geometry.

### `results/`

Contains generated results, figures, optimization histories, and summaries.

The optimization script may generate local temporary FDTD session files under a `temp_sessions/` directory. These files are intended for debugging and are not part of the published repository.

### `scripts/`

Contains the main user-facing Python scripts.

### `src/`

This directory is currently reserved for future code organization. Shared functions may be moved here later as the repository develops.

## Requirements

The analytical and data-processing scripts use standard scientific Python packages, including:

- NumPy
- SciPy
- scikit-learn
- joblib
- Matplotlib

The photonic optimization workflow additionally requires:

- Lumerical FDTD
- `lumapi`
- LumOpt
- A valid local Lumerical Python API installation

The optimization script expects the Lumerical Python API path to be available through the `LUMERICAL_API_PATH` environment variable. If this variable is not set, the script uses its configured default path.

Example for Windows PowerShell:

```powershell
$env:LUMERICAL_API_PATH = "C:\Program Files\Lumerical\v252\api\python"
```

## Installation

Clone the repository:

```bash
git clone [https://github.com/elsaym15/pca-coeff-opt.git](https://github.com/elsaym15/pca-coeff-opt.git)
cd pca-coeff-opt
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the Python dependencies:

```bash
pip install numpy scipy scikit-learn joblib matplotlib
```

Lumerical FDTD, `lumapi`, and LumOpt must be installed and configured separately.

## Workflow overview

The recommended photonic workflow is:

```text
Generate photonic devices
          |
          v
Fit PCA model and coefficients
          |
          v
Validate Lumerical monitor dimensions
          |
          v
Run PCA-coefficient optimization
          |
          v
Inspect results and saved diagnostics
```

## Running the analytical example

The analytical example is independent of Lumerical and can be used to test PCA-reduced optimization in a controlled setting.

Run:

```bash
python scripts/run_analytical_example.py
```

The generated figures and numerical results should be written to:

```text
results/analytical/
```

## Generating photonic devices

Run the device-generation script:

```bash
python scripts/generate_photonic_devices.py
```

This script creates or prepares the photonic device ensemble used for PCA analysis.

The generated data may be written to the `data/raw/` or `data/processed/` directories depending on the configuration in the script.

## Fitting the photonic PCA model

After generating the device ensemble, run:

```bash
python scripts/fit_photonic_pca.py
```

This script fits the PCA model and saves the main PCA artifacts in:

```text
data/models/
```

The optimization workflow expects the following files:

```text
data/models/photonic_pca.joblib
data/models/photonic_coefficients.npy
```

The PCA analysis may also produce variance-analysis files and reconstructed ensembles.

## Running the photonic optimization

Run the optimization script from the repository root:

```bash
python scripts/optimize_photonic_device.py --device-index 1 --method SLSQP
```

The `--device-index` argument selects the row of `photonic_coefficients.npy` used as the initial design.

For example:

```bash
python scripts/optimize_photonic_device.py --device-index 0 --method SLSQP
```

A custom run name can be supplied for debugging:

```bash
python scripts/optimize_photonic_device.py \
    --device-index 1 \
    --method SLSQP \
    --run-name test_run_01
```

On Windows PowerShell, the same command can be written as:

```powershell
python scripts/optimize_photonic_device.py `
    --device-index 1 `
    --method SLSQP `
    --run-name test_run_01
```

## What the optimization script does

The optimization script performs the following operations:

1. Loads `photonic_pca.joblib`.
2. Loads `photonic_coefficients.npy`.
3. Evaluates `optimization_base_run.lsf`.
4. Retrieves the `opt_fields` monitor coordinates.
5. Checks that the monitor-grid size matches the number of PCA features.
6. Selects the initial PCA coefficient vector.
7. Loads `optimization_base_norun.lsf` for each optimization evaluation.
8. Reconstructs the photonic geometry from PCA coefficients.
9. Injects the reconstructed geometry into the Lumerical simulation.
10. Runs the forward simulation.
11. Runs the adjoint simulation.
12. Computes the gradient in pixel space.
13. Projects the gradient into PCA-coefficient space.
14. Passes the objective and gradient to the SciPy optimizer.
15. Saves optimization results, logs, and diagnostic files.

## Monitor-grid validation

The PCA model and Lumerical simulation must use compatible dimensions.

If the PCA model has \(N\) features and the Lumerical monitor has dimensions \(N_x\) and \(N_y\), the following condition must hold:

```text
N = N_x × N_y
```

The optimization script checks this before starting the optimization. If the dimensions do not match, the script stops and reports the mismatch.

## Generated outputs

A typical optimization run creates a directory similar to:

```text
results/photonics/optimization/
└── device_1_slsqp/
    ├── initial_coefficients.npy
    ├── final_coefficients.npy
    ├── final_index_map.npy
    ├── final_epsilon_map.npy
    ├── history.json
    └── run_summary.json
```

Debugging logs are stored under:

```text
results/photonics/optimization/logs/
```

Temporary forward and adjoint FDTD sessions are stored locally under:

```text
results/photonics/optimization/temp_sessions/
```

These `.fsp` files are useful for inspecting individual optimization evaluations in Lumerical, but they are intentionally excluded from the public repository.

## Debugging

The optimization script records information about:

- Loaded PCA and coefficient files.
- Lumerical API initialization.
- Monitor dimensions.
- PCA feature count.
- Reconstructed index-map shape.
- Forward and adjoint simulation sessions.
- Figure-of-merit values.
- Transmission values.
- Projected-gradient norms.
- Sensitivity-map dimensions.
- Optimization status and termination messages.

If an optimization run fails, inspect the corresponding log file in:

```text
results/photonics/optimization/logs/
```

If a specific FDTD evaluation needs to be inspected, open the corresponding `.fsp` file from the run's `temp_sessions/` directory in Lumerical.

## Important notes

- Lumerical FDTD and LumOpt are external requirements and are not included in this repository.
- The optimization workflow currently depends on the monitor names and simulation objects defined in the Lumerical scripts.
- The PCA feature count must match the dimensions of the Lumerical `opt_fields` monitor.
- The `src/` directory is currently empty or reserved for future refactoring.
- Temporary `.fsp` simulation files are generated locally and are not included in the published repository.
- Paths in the scripts may need to be adjusted for a different operating system or Lumerical installation.
- The repository is an actively developing research codebase and may receive further organization and modularization in future updates.

## Future development

Planned improvements may include:

- Moving reusable functions into `src/`.
- Adding automated tests for PCA reconstruction and gradient projection.
- Adding a configuration file for simulation and optimization parameters.
- Adding automated post-processing scripts.
- Adding reproducibility examples with smaller test datasets.
- Adding continuous-integration checks for scripts that do not require Lumerical.

## License

Add the project license here before distributing the repository publicly.
