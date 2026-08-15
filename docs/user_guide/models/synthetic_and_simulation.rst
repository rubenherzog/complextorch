Synthetic systems and trajectory simulation
===========================================

ComplexTorch separates **construction of dynamical systems** from **simulation
of observations from those systems**.  This distinction mirrors the source
architecture: synthetic model and covariance constructors live in
``complextorch.synthetic``, while ``complextorch.simulate`` is responsible for
trajectory generation.

Synthetic model construction
----------------------------

:func:`~complextorch.synthetic_var` constructs canonical batched
:class:`~complextorch.VARSystem` objects for controlled parameter sweeps.  The
related public helpers
:func:`~complextorch.available_synthetic_systems`,
:func:`~complextorch.synthetic_system_parameters`,
:func:`~complextorch.synthetic_transition_matrix`,
:func:`~complextorch.equicorrelated_innovation_covariance`, and
:func:`~complextorch.planted_module_projection` describe or construct the
corresponding model ingredients.

Additional constructors are also part of the synthetic layer:

- :func:`~complextorch.demo_var` builds the historical demonstration VAR;
- :func:`~complextorch.random_stable_var` generates random stable VAR
  coefficients and innovation covariance;
- :func:`~complextorch.random_correlation_matrix` generates correlation
  matrices with the onion construction corresponding to LKJ ``eta=1``;
- :func:`~complextorch.random_positive_definite_covariance` combines those
  correlation matrices with random marginal scales.

The random-correlation implementation is Torch-first and uses a local
:class:`torch.Generator`, so dtype and device are preserved by construction.
These functions create model parameters or canonical models; they do not create
time-series observations.

Trajectory simulation
---------------------

:func:`~complextorch.simulate_var` takes stable VAR coefficients and a positive
definite innovation covariance and generates independent Gaussian VAR
trajectories,

.. math::

   x_t=\sum_{k=1}^{p}A_kx_{t-k}+\varepsilon_t,
   \qquad
   \varepsilon_t\sim\mathcal N(0,\Sigma).

Input coefficients may have shape ``(p, n, n)`` or
``(batch, p, n, n)``.  The returned observations have shape
``(batch, time, n)``.  Every batch item is simulated independently: no state or
lag is propagated across batch boundaries.

The ``burnin`` argument can be a non-negative integer or ``"auto"``.
:func:`~complextorch.automatic_burnin` chooses a shared burn-in length from the
largest companion spectral radius in the supplied batch.  With
``return_innovations=True``, :func:`~complextorch.simulate_var` also returns the
retained innovation sequence with the same batch/time/variable shape.

Construction followed by simulation
-----------------------------------

A synthetic model can therefore be constructed and simulated as two explicit
steps:

.. code-block:: python

   from complextorch import simulate_var, synthetic_var

   system = synthetic_var(
       "directed_ring",
       6,
       spectral_radius_target=0.8,
   )

   observations = simulate_var(
       system.coefficients,
       system.innovation_covariance,
       1000,
       burnin="auto",
       seed=0,
   )

Keeping these operations separate is useful when the scientific analysis works
directly with a supplied or synthetic dynamical model: model-derived measures
can be evaluated without simulating observations.  Simulation is only needed
when an explicit trajectory is part of the experiment.

Compatibility note
------------------

The repository cleanup moved synthetic model/covariance ownership to
``synthetic.py`` and kept ``simulate.py`` focused on trajectories.  Historical
module-level imports of ``demo_var`` and the random VAR/covariance helpers from
``complextorch.simulate`` remain re-exported for compatibility, while new code
should use the public top-level ComplexTorch API.
