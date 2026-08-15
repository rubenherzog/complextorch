Confidence intervals for model-derived measures
===============================================

:func:`~complextorch.measure_confidence_intervals` estimates sampling
uncertainty for analytical measures derived from a fixed-order VAR model. The
procedure follows the resampling logic of Beda, Simpson, and Faes (2017): fit an
autoregressive model, generate surrogate realizations, refit the model, and use
the empirical distribution of the derived index to obtain confidence limits.
ComplexTorch applies this principle to its multivariate VAR representation and
evaluates all requested compatible measures on one shared resampling ensemble.

Scientific question
-------------------

The interval answers: *how much would this model-derived measure vary under
repeated finite-sample realizations of the fitted VAR process?* It does not test
against a structural null model and it does not perform model-order selection.
The VAR order supplied to the estimator is held fixed throughout resampling.

Resampling procedures
---------------------

Two procedures are available through the ``method`` argument:

``"residual_bootstrap"``
   Resamples fitted residual vectors and propagates them through the fitted VAR.

``"parametric"``
   Draws Gaussian innovations from the fitted innovation covariance and
   propagates them through the fitted VAR.

For each surrogate dataset ComplexTorch refits the same fixed-order,
unregularized OLS VAR. Unstable resampled models are discarded. Central
percentile limits are then computed from the remaining stable ensemble. For a
confidence level :math:`1-\alpha`, the reported limits are the empirical
:math:`\alpha/2` and :math:`1-\alpha/2` quantiles.

Shared-ensemble contract
------------------------

The resampling ensemble is constructed once. All configured compatible measures
are evaluated on that same ensemble through
:class:`~complextorch.InferenceMeasureConfig`; requesting additional compatible
measures therefore does not trigger an independent bootstrap for each measure.
This preserves a common resampling axis across quantities that may later be
compared.

Dynamical dependence is evaluated only for a supplied fixed macro projection or
fixed batch of projections. Projection optimization is intentionally not
repeated inside bootstrap replicates.

Batch semantics
---------------

Input observations follow the standard ComplexTorch convention:
``(time, variables)`` or ``(batch, time, variables)``. In ``mode="pooled"``
trajectory boundaries are preserved when constructing lagged regressions. In
``mode="independent"`` each trajectory is refitted independently; a bootstrap
replicate is retained only when all trajectory-specific VAR systems required for
that replicate are stable.

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
