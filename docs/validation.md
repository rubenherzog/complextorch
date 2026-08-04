# Validation

The initial validation ladder is intentionally redundant:

1. Estimator parity with `statsmodels.VAR` in float64.
2. Batched fitting parity with a loop of single-epoch fits.
3. Explicit companion-matrix layout tests.
4. Torch Lyapunov doubling/direct parity with SciPy.
5. Analytical CMem parity with an independent NumPy/SciPy reference.
6. CMem chain-rule identities.
7. Recovery of generating VAR coefficients from long simulations.
8. Recovery of the correct VAR order by temporal cross-validation.
9. CPU/CUDA parity when CUDA is available.

The development environment used for the initial validation had CPU-only PyTorch, so CUDA coverage is included as a conditional test and is not presented as an executed GPU benchmark.
