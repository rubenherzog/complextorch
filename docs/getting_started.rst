Getting started
===============

Installation
------------

ComplexTorch requires Python 3.10 or newer. For development, install the package
with its development dependencies::

   python -m pip install -e ".[dev]"

Minimal example
---------------

The example below uses :func:`~complextorch.demo_var` to construct a stable
system, :func:`~complextorch.simulate_var` to generate observations,
:class:`~complextorch.VAR` for estimation, and
:func:`~complextorch.temporal_mvgc` / :func:`~complextorch.spectral_mvgc` for
Granger-causality analysis. Each linked API page includes a ``[source]`` link to
the implementation.

.. code-block:: python

   import torch

   from complextorch import VAR, demo_var, simulate_var
   from complextorch import spectral_mvgc, temporal_mvgc

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

Time-series estimators such as :class:`~complextorch.VAR`,
:class:`~complextorch.N4SID`, and :class:`~complextorch.LarimoreStateSpace`
accept either ``(time, variables)`` or ``(batch, time, variables)`` inputs where
documented. Independent trajectories must remain separated; batching must not
create lags, transitions, residual pairs, or validation links across trajectory
boundaries.

Dynamical-independence workflow
-------------------------------

For dynamical dependence / SSDI, :func:`~complextorch.optimise_dynamical_dependence`
now uses the validated staged SSDI workflow by default when ``objective=None``:
proxy-DD pre-optimization over many random restart subspaces, Grassmann
clustering of proxy minima, and spectral-DD refinement of cluster
representatives. The result is a :class:`~complextorch.DDSSDIOptimizationResult`.
Explicit ``objective="proxy"`` or ``objective="spectral"`` requests retain the
single-stage research API and return :class:`~complextorch.DDOptimizationResult`.

Next steps
----------

Continue with the :doc:`user_guide/index` for mathematical and scientific
conventions, the :doc:`auto_examples/index` for executable examples, and the
:doc:`api` for the complete linked public API.
