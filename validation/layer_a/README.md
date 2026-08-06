# Layer A parity validation

These executable Python scripts validate the numerical foundations that later fitting and MVGC notebooks depend on. Notebooks are intentionally excluded until every script has run successfully and the diagnostics have been reviewed.

The comparisons use three references where available:

1. independent analytical formulas or SciPy solvers;
2. ComplexBox pinned to commit `87b5e2cd9bba22ddd978bade6f614da7d6190db2`;
3. the current ComplexTorch checkout.

The suite distinguishes exact representation parity, numerical parity, equivalent observable representations, expected convention differences, and actual failures. It never enlarges tolerances merely to force agreement.

Run all scripts from the repository root:

```bash
python validation/layer_a/run_all.py --output validation/layer_a/results
```

Generated outputs include per-suite JSON and CSV files, an aggregated summary, exact synthetic-system arrays, environment metadata, and execution return codes.

Current scripts:

- `check_environment_conventions.py`: axis layouts, adapters, dtype, and VAR companion conventions.
- `check_synthetic_ground_truth.py`: fixed stable VAR and SSM systems, covariance validity, analytical stationarity, observability, and Monte Carlo sanity checks.
- `check_dare_lyapunov.py`: direct/doubling Lyapunov solvers, standard DARE, generalized DARE, and equation residuals.
- `check_var_ss_conversions.py`: VAR companion and innovations representations plus general SSM-to-innovations conversion.
- `check_autocovariances.py`: VAR and SSM autocovariances, Yule-Walker recursion, representation invariance, and block-Toeplitz positivity.
- `check_transfer_spectra.py`: VAR/SS transfer functions, inverse transfer identity, cross-spectral density, Hermitian/PSD structure, and Wiener-Khinchin recovery of lag-zero covariance.
