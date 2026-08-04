# ComplexTorch

Torch-first batched inference of vector autoregressive models and analytical dynamical measures for neural time series.

## Core architecture

- NumPy or Torch input, normalized internally to `(batch, time, variables)`.
- Independent per-epoch or pooled VAR fitting.
- OLS and Ridge estimation using batched linear algebra.
- scikit-learn-compatible estimator surface.
- Epoch-aware temporal cross-validation for model-order selection.
- Exact VAR companion state-space representation.
- Batched stationary covariance through the discrete Lyapunov equation.
- Stable random and structured cyclic/frustrated synthetic VAR generators.

## Measures

All public measures are exposed from `complextorch.measures` and listed in `MEASURE_REGISTRY`.

### Gaussian information

- Gaussian entropy.
- Mutual information and conditional mutual information.
- Conditional covariance.
- Total correlation (TC), dual total correlation (DTC), O-information and S-information.
- Local Gaussian mutual information.

### Time and frequency domain

- VAR autocovariances.
- Entropy rate.
- Predictive information.
- Per-variable active information storage.
- Transfer and inverse-transfer functions.
- Cross-spectral density.
- Spectral entropy.

### Dynamical diagnostics

- Spectral radius.
- Stability margin.
- Dominant timescale.
- Covariance amplification.

### Emergence

- Ψ, Δ and Γ from a fitted `VARSystem` and a linear macro projection.
- Observational Gaussian plug-in calculation for Ψ.

### CMem

- CMem1 and CMem3 totals.
- CMem1 and CMem3 lag curves.
- Conditional-lag CMem3 decomposition.

## Installation

```bash
python -m pip install -e ".[dev]"
```

## Example

```python
import torch
from complextorch import VAR, demo_var, simulate_var
from complextorch.measures import DynamicalMeasures

coef, noise = demo_var(n_variables=3, order=2)
X = simulate_var(coef, noise, n_times=4000, seed=1)
model = VAR(order=2, mode="independent").fit(X)
system = model.to_var_system()

frequencies = torch.linspace(0, 0.5, 128, dtype=system.coefficients.dtype)
macro = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=system.coefficients.dtype)

values = DynamicalMeasures(
    [
        "spectral_radius",
        "predictive_information",
        "cross_spectral_density",
        "psi",
        "cmem3_total",
    ],
    frequencies=frequencies,
    macro_projection=macro,
)(system)
```

General latent-state SSM identification, Kalman filtering/smoothing, N4SID, EM, MVGC, PhiID and SSDI remain planned work.
