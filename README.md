# ComplexTorch

Torch-first batched inference of vector autoregressive models and analytical dynamical measures for neural time series.

## Implemented

- NumPy or Torch input, normalized internally to `(batch, time, variables)`.
- Independent per-epoch or pooled VAR fitting.
- OLS and Ridge estimation using batched linear algebra.
- scikit-learn-compatible estimator surface.
- Epoch-aware temporal cross-validation for model-order selection.
- Exact VAR companion state-space representation.
- Batched stationary covariance through the discrete Lyapunov equation.
- Gaussian information primitives, criticality diagnostics and CMem measures.
- Stable random and structured cyclic/frustrated synthetic VAR generators.

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
values = DynamicalMeasures([
    "spectral_radius",
    "dominant_timescale",
    "cmem3_total",
])(system)
```

The current version is an initial validated VAR-first architecture. General latent-state SSM identification, Kalman filtering/smoothing, N4SID, EM, MVGC, PhiID and SSDI remain planned work.
