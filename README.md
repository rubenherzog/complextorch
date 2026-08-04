# ComplexTorch

Torch-first batched inference of vector autoregressive and latent linear state-space models, with analytical and sample-based complexity measures.

## Design

ComplexTorch uses shared computational primitives rather than separate implementations for each measure:

- `measures/_model_comparison.py` owns variable indexing, full/reduced VAR fitting, conditional covariance and spectral comparisons, and log-determinant ratios.
- `control.py` owns DARE/Riccati, innovations form, observational reduction and linear projections used by Kalman, dynamical dependence and SSDI.
- `measures/gaussian.py` is the single Gaussian entropy and mutual-information implementation used across emergence and PhiID.
- `measures/phid.py` builds the complete 16-atom bivariate Gaussian MMI double-redundancy lattice using generic Möbius inversion.

## Implemented coverage

- Batched VAR fitting from NumPy or Torch, with independent or pooled epochs.
- Temporal cross-validation for model-order selection.
- Companion and latent state-space representations.
- Kalman filtering and smoothing, N4SID and linear-Gaussian EM.
- DARE/Riccati, model reduction, innovations form and projection search.
- Group temporal and spectral MVGC plus bivariate Geweke spectral GC.
- Dynamical dependence and SSDI/stochastic interaction.
- Complete 16-atom Gaussian MMI PhiID and aggregate compatibility API.
- Gaussian information measures, emergence measures, spectra, criticality and CMem.
- Discrete entropy, mutual information, total correlation and LZ76 complexity.

## Installation

```bash
python -m pip install -e ".[dev]"
```

## Example

```python
import torch
from complextorch import VAR, demo_var, simulate_var
from complextorch.measures import temporal_mvgc, spectral_mvgc

coefficients, noise = demo_var(n_variables=3, order=2)
data = simulate_var(coefficients, noise, n_times=4000, seed=1)
model = VAR(order=2).fit(data)

gc_time = temporal_mvgc(data, 2, source=[1], target=[0], conditional=[2])
gc_frequency = spectral_mvgc(
    data,
    2,
    source=[1],
    target=[0],
    conditional=[2],
    frequencies=torch.linspace(0, 0.5, 128, dtype=torch.float64),
)
```

Current package version: **0.3.0**.
