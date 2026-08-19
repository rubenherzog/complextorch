Confidence intervals for model-derived measures
===============================================

:func:`~complextorch.measure_confidence_intervals` estimates sampling
uncertainty for analytical measures derived from a fixed-complexity linear
Gaussian model. The supported fitting routes are fixed-order VAR, general
state-space, and innovations-form state-space estimators. The procedure is the
same in each case: fit the model, generate surrogate realizations from that
fitted observable process, refit the same estimator family without repeating
model selection, and obtain percentile limits from the empirical distribution
of the derived measure.

Scientific question
-------------------

The interval answers: *how much would this model-derived measure vary under
repeated finite-sample realizations of the fitted process?* It does not test
against a structural null model. VAR order, latent state dimension, subspace
horizons, and other estimator-complexity choices are held fixed throughout
resampling.

Resampling procedures
---------------------

Two procedures are available through the ``method`` argument:

``"residual_bootstrap"``
   Resamples complete fitted one-step innovation vectors, preserving their
   contemporaneous multivariate dependence.

``"parametric"``
   Draws Gaussian innovation vectors from the fitted innovation covariance.

VAR surrogates retain the existing direct AR recursion. State-space systems are
converted exactly to the common steady-state innovations representation

.. math::

   z_{t+1}=Az_t+K\varepsilon_t,\qquad
   y_t=Cz_t+\varepsilon_t,

with :math:`\operatorname{cov}(\varepsilon_t)=V`. The same recursion is used
for both general :class:`~complextorch.StateSpaceModel` and
:class:`~complextorch.InnovationsStateSpace` fits, so simulation logic is not
duplicated across representations.

Each surrogate is refitted with the same fixed-complexity estimator. Unstable
refits are discarded. For confidence level :math:`1-\alpha`, the reported
limits are the empirical :math:`\alpha/2` and :math:`1-\alpha/2` quantiles.

Estimator choice
----------------

The ``estimator`` argument accepts :class:`~complextorch.VAR`,
:class:`~complextorch.N4SID`, :class:`~complextorch.LarimoreStateSpace`, or
:class:`~complextorch.LinearGaussianEM`. The older ``var=`` argument remains a
backward-compatible alias for VAR inference.

The bootstrap distribution is estimator-specific. In numerical validation, a
bare compact N4SID fit was not self-centering for information-theoretic
functionals of a known VAR process, whereas N4SID followed by fixed-iteration
:class:`~complextorch.LinearGaussianEM` refinement recovered intervals
compatible with the VAR and Larimore innovations-form routes. For inferential
work with a general state-space representation, EM refinement is therefore the
validated route rather than assuming that subspace identification alone
provides a calibrated stochastic model.

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
trajectory boundaries are preserved and no state transition, lag, or residual
pair crosses between trajectories. In ``mode="independent"`` each trajectory is
refitted independently; a bootstrap replicate is retained only when all
trajectory-specific fitted systems required for that replicate are stable.

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
