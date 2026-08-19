Confidence intervals for model-derived measures
===============================================

:func:`~complextorch.measure_confidence_intervals` estimates sampling
uncertainty for analytical measures derived from a fixed-complexity linear
Gaussian model. Supported fitting routes are fixed-order VAR, innovations-form
state space (Larimore/CVA), and general state space (N4SID or
:class:`~complextorch.LinearGaussianEM`). The estimator is fitted once, one
shared surrogate ensemble is generated, the same fixed-complexity estimator is
refitted, and all requested compatible model-derived measures are evaluated on
that ensemble.

Scientific question
-------------------

The interval answers: *how much would this model-derived measure vary under
repeated finite-sample realizations of the fitted process?* It is sampling
uncertainty, not a structural null test. VAR order, latent dimension, subspace
horizons, EM iterations, and other model-complexity choices are held fixed
throughout resampling; model selection is not repeated inside bootstrap.

Resampling procedures
---------------------

Two procedures are available through ``method``:

``"residual_bootstrap"``
   Resamples complete fitted one-step innovation vectors. The vector is the
   sampling unit, so contemporaneous multivariate dependence is preserved.

``"parametric"``
   Draws Gaussian innovation vectors from the fitted innovation covariance.

VAR keeps the existing direct AR recursion and batched OLS refit path. All
state-space estimators instead share the canonical innovations representation

.. math::

   z_{t+1}=Az_t+K\varepsilon_t,\qquad
   y_t=Cz_t+\varepsilon_t,

with :math:`\operatorname{cov}(\varepsilon_t)=V`. General state-space models are
converted through :func:`~complextorch.as_innovations_state_space`; an already
fitted :class:`~complextorch.InnovationsStateSpace` needs no representation
change. This common path is used for surrogate simulation rather than
maintaining separate state-space simulation kernels.

Estimator choice
----------------

The ``estimator`` argument accepts :class:`~complextorch.VAR`,
:class:`~complextorch.LarimoreStateSpace`, :class:`~complextorch.N4SID`, or
:class:`~complextorch.LinearGaussianEM`. The historical ``var=`` argument
remains a backward-compatible alias for VAR inference.

Larimore is the direct innovations-form identification route and therefore the
most direct state-space path to the common analytical backbone used by many
ComplexTorch measures. N4SID is also supported, but its bootstrap distribution
reflects the fitted N4SID stochastic model; subspace identification alone should
not be assumed to have the same finite-sample calibration as VAR or a refined
state-space likelihood fit. ``LinearGaussianEM`` retains the supplied fitted
system as the initialization for every bootstrap refit.

Shared-ensemble and measure contract
------------------------------------

The resampling ensemble is constructed once. Measure evaluation delegates to
the same canonical :func:`~complextorch.compute_all_model_measures` machinery
used outside inference, with inference-specific aliases retained for backward
compatibility. Requesting additional compatible measures therefore does not
trigger additional simulation or refitting. This includes scalar, matrix,
spectral, information-rate, criticality, CMem, MVGC, and configured higher-order
families such as PIRD/PDGC/HOP when their required structural configuration is
supplied.

Dynamical dependence is evaluated only for a supplied fixed macro projection or
fixed batch of projections. Projection optimization is intentionally not
repeated inside bootstrap replicates.

Batch semantics
---------------

Inputs follow the standard ComplexTorch convention: ``(time, variables)`` or
``(batch, time, variables)``. In ``mode="pooled"`` a common model is estimated
across independent trajectories while preserving every trajectory boundary. No
lag, state transition, Hankel relation, or residual pair crosses a boundary. In
``mode="independent"`` one model is estimated per trajectory; a bootstrap
replicate is retained only when every trajectory-specific refit required for
that replicate is stable, preserving a common bootstrap axis across measures.

Main API
--------

- :func:`~complextorch.measure_confidence_intervals`
- :class:`~complextorch.InferenceMeasureConfig`
- :class:`~complextorch.ConfidenceIntervalResult`
- :class:`~complextorch.MeasureInterval`

Reference
---------

Beda, A., Simpson, D. M., & Faes, L. (2017). *Estimation of confidence limits
for descriptive indexes derived from autoregressive analysis of time series:
Methods and application to heart rate variability*. PLOS ONE, 12(10), e0183230.
https://doi.org/10.1371/journal.pone.0183230
