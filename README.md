# ComplexTorch

ComplexTorch is a **Torch-first scientific toolbox for the analysis of complex multivariate dynamical systems** using vector autoregressive (VAR) and linear state-space models.

It provides a unified model-based framework for studying **complexity, criticality, information processing, causal interactions, and emergence**, deriving these quantities from a common underlying dynamical model whenever possible. This makes otherwise heterogeneous measures easier to compare, combine, and apply consistently to the same system.

Built on PyTorch, ComplexTorch is designed for **efficient, scalable scientific computation**, with batched numerical operations and CPU/GPU acceleration for analyzing large systems and parameter sweeps.

**[Documentation → Read the Docs](https://complexbox-torch.readthedocs.io/en/latest/)**

## Highlights

- **Unified model-based analysis** of complexity, information processing, causality, criticality, and emergence.
- VAR and linear state-space modeling, simulation, model selection, and diagnostics.
- Information-theoretic measures including **TC, DTC, O-information, S-information, MIR, TE, PhiID, O-information rate, PIRD, PDGC**, and higher-order measures.
- Temporal and spectral **multivariate Granger causality**.
- Measures of **dynamical dependence, stochastic interaction, predictive emergence, and criticality**.
- Synthetic dynamical systems and reproducible examples for scientific validation and exploration.

## Installation

```bash
pip install complextorch
```

For development:

```bash
python -m pip install -e ".[dev]"
```

## Quick example

```python
from complextorch import VAR, demo_var, simulate_var

coefficients, noise = demo_var(n_variables=3, order=2)
data = simulate_var(coefficients, noise, n_times=4000, seed=1)

model = VAR(order=2).fit(data)
```

See the **[documentation](https://complexbox-torch.readthedocs.io/en/latest/)** for the scientific user guide, mathematical definitions, API reference, tutorials, and executable examples.

## Authors

ComplexTorch is developed by [Rubén Herzog](https://github.com/rubenherzog) and [Boki Milinkovic](https://github.com/bmilinkovic).

See the documentation for citation information.
