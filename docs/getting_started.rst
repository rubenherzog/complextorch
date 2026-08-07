Getting started
===============

Installation
------------

ComplexTorch requires Python 3.10 or newer. For development, install the package
with its development dependencies::

   python -m pip install -e ".[dev]"

Minimal example
---------------

.. code-block:: python

   import torch

   from complextorch import VAR, demo_var, simulate_var
   from complextorch.measures import spectral_mvgc, temporal_mvgc

   coefficients, noise = demo_var(n_variables=3, order=2)
   data = simulate_var(coefficients, noise, n_times=4000, seed=1)
   model = VAR(order=2).fit(data)

   gc_time = temporal_mvgc(
       data,
       2,
       source=[1],
       target=[0],
       conditional=[2],
   )
   gc_frequency = spectral_mvgc(
       data,
       2,
       source=[1],
       target=[0],
       conditional=[2],
       frequencies=torch.linspace(0, 0.5, 128, dtype=torch.float64),
   )

Batch convention
----------------

Time-series estimators accept either ``(time, variables)`` or
``(batch, time, variables)`` inputs where documented. Independent trajectories
must remain separated; batching must not create lags, transitions, residual
pairs, or validation links across trajectory boundaries.

The detailed scientific user guide, mathematical conventions, and worked
examples are intentionally reserved for the next documentation phase. Phase 1
establishes the build and API-reference infrastructure only.
