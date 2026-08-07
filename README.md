# ComplexTorch

Torch-first batched inference of vector autoregressive and latent linear state-space models, with model-derived analytical complexity and information measures.

## Design

ComplexTorch treats mathematical and numerical contracts as part of the public API. Shared primitives are reused across estimators and measures rather than reimplemented per analysis:

- `representations.py` defines the canonical `VARSystem` and `StateSpaceModel` representations.
- `control.py` owns DARE/Riccati solvers, innovations-form conversion, exact observational reduction, and linear projections used by state-space MVGC and dynamical dependence.
- `selection/` separates VAR lag selection from state-space latent-dimension selection and uses temporal rather than shuffled cross-validation.
- `measures/primary.py` provides the model-first analytical layer and shared model-measure context.
- `measures/gaussian.py`, `rates.py`, `oir.py`, `pird.py`, `pdgc.py`, and `hop.py` provide Gaussian information, rate, high-order, and decomposition primitives.
- `measures/phid.py` and `measures/phid_primary.py` provide bivariate Gaussian PhiID and model-derived redundancy backends.
- `dd.py` exposes the public dynamical-dependence optimizer; `dd_ssdi.py` orchestrates the validated staged SSDI workflow without duplicating the proxy/spectral objective kernels.

## Implemented coverage

- Batched VAR fitting from NumPy or Torch, with independent or pooled trajectories.
- VAR information-criterion selection and temporal cross-validation.
- Canonical VAR, general state-space, and innovations-form representations.
- Kalman filtering and smoothing, N4SID, Larimore/CVA state-space fitting, and linear-Gaussian EM.
- Standard and generalized DARE/Riccati solvers, exact model reduction, innovations form, and projections.
- Group temporal and spectral MVGC, Gaussian MIR/TE rates, O-information rate, PIRD, PDGC, and HOP analyses.
- Gaussian TC, DTC, O-information, S-information, PhiID, predictive emergence, criticality, spectra, and related model-derived measures.
- Dynamical dependence and SSDI with staged proxy multi-start optimization, Grassmann clustering, and spectral refinement by default.
- ComplexBox-compatible and native Riemannian Armijo DD optimizer backends under one public contract.
- Deterministic simulation helpers and executable validation/examples.

## Documentation

The Sphinx/ReadTheDocs source is under [`docs/`](docs/index.rst). The documentation includes a scientific user guide, executable Sphinx-Gallery examples, and generated per-object API pages with `[source]` links back to the implementation.

The dedicated dynamical-dependence guide documents the current staged SSDI default, result contracts, Grassmann geometry, and reproducibility requirements.

## Installation

```bash
python -m pip install -e ".[dev]"
```

Documentation dependencies can be installed with:

```bash
python -m pip install -e ".[docs]"
```

## Example

```python
import torch
from complextorch import VAR, demo_var, simulate_var
from complextorch import spectral_mvgc, temporal_mvgc

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

## Dynamical dependence / SSDI

`optimise_dynamical_dependence(...)` uses the staged SSDI workflow when `objective=None` (the default): proxy-DD optimization over many restart subspaces, SSDI/ComplexBox-compatible Grassmann clustering, and full spectral-DD refinement from one representative per cluster. Explicit `objective="proxy"` or `objective="spectral"` preserves the single-stage research API.

Current package version: **0.8.0**.
