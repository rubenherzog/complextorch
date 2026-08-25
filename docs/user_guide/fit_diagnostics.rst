Fit diagnostics
===============

ComplexTorch separates predictive accuracy, temporal residual adequacy, and
probabilistic calibration. These are complementary questions: a model can
predict well on average while leaving temporal structure in its residuals, or
produce unbiased predictions with miscalibrated predictive uncertainty.

:func:`~complextorch.fit_diagnostics` supports two explicit evaluation regimes
for a fitted :class:`~complextorch.VAR`, :class:`~complextorch.LarimoreStateSpace`,
:class:`~complextorch.N4SID`, or :class:`~complextorch.LinearGaussianEM`:
``evaluation="oos"`` evaluates a chronologically later test block using rolling
one-step-ahead prediction, whereas ``evaluation="in_sample"`` evaluates
one-step residuals on the supplied fitted-data trajectory. The estimator is
never refitted by the diagnostic. Both regimes share the same lower-level
:func:`~complextorch.innovation_diagnostics` implementation.

For observations :math:`y_t` and one-step predictions
:math:`\hat y_{t\mid t-1}`, define

.. math::

   e_t = y_t-\hat y_{t\mid t-1}.

Three complementary diagnostic axes
-----------------------------------

A useful default interpretation is to read the output along three axes rather
than to search for a single universal goodness-of-fit score.

.. list-table::
   :header-rows: 1
   :widths: 28 34 38

   * - Question
     - Primary diagnostics
     - Interpretation
   * - Does the model predict accurately?
     - ``nmse``, ``predictive_r2``, ``rmse``
     - Smaller NMSE/RMSE and larger predictive :math:`R^2` are better.
   * - Are one-step errors temporally structureless?
     - ``whiteness_energy``
     - Smaller is better; zero means no detected lagged linear dependence
       through ``max_lag``.
   * - Is predictive uncertainty calibrated?
     - ``covariance_calibration``, ``prediction_interval_coverage``
     - Smaller covariance-calibration error is better; marginal interval
       coverage should be close to its nominal Gaussian level.

``gaussian_nll`` combines prediction error and predictive covariance and is
therefore useful as an overall probabilistic score, but it should not replace
inspection of the three axes above. ``consistency`` and the additional residual
statistics described below are complementary diagnostics rather than a fourth
primary axis.

Predictive accuracy
-------------------

The result reports RMSE, Gaussian negative log likelihood, normalized mean
squared error,

.. math::

   \operatorname{NMSE}
   = \frac{\sum_t\lVert e_t\rVert_2^2}
           {\sum_t\lVert y_t-\bar y_{\mathrm{train}}\rVert_2^2},

and the corresponding predictive coefficient
:math:`R^2_{\mathrm{pred}}=1-\operatorname{NMSE}`. NMSE and predictive
:math:`R^2` contain the same information with opposite orientation, so they
should not be treated as independent diagnostics.

The one-step predictive mean is also exposed as ``prediction_mean``. This is
the same prediction used to construct the residuals; no second prediction
recursion is run for reporting.

Probabilistic calibration
-------------------------

Let :math:`V` denote the predictive innovation covariance estimated from the
fitted model. ``prediction_covariance`` exposes this covariance directly.
Errors are standardized by a Cholesky factor :math:`V=LL^\top`,

.. math::

   z_t=L^{-1}e_t.

A calibrated innovations model should have approximately zero standardized mean
and identity standardized covariance. ``standardized_errors`` exposes the
individual :math:`z_t`, while ``covariance_calibration`` is

.. math::

   \frac{\lVert\widehat\Sigma_z-I\rVert_F}{\sqrt n},

so smaller values indicate better covariance calibration.

``prediction_interval_coverage`` reports the empirical marginal coverage of
nominal 95% Gaussian one-step predictive intervals constructed from the same
predictive mean and covariance. Values near 0.95 are expected under a calibrated
Gaussian predictive model. Coverage is a diagnostic, not a binary acceptance
test: finite samples, non-Gaussian innovations, and dependence between
coordinates can all move the empirical value away from 0.95.

Temporal residual adequacy
--------------------------

Temporal adequacy is assessed from lagged autocorrelation and cross-correlation
matrices of the standardized one-step errors. The sample lag matrices are
normalized by the held-out residual covariance so that temporal dependence is
separated from covariance-calibration error. The primary model-comparison score
is

.. math::

   W_h=\sum_{k=1}^{h}\lVert R_z(k)\rVert_F^2.

``whiteness_energy`` therefore has a direct interpretation: zero corresponds to
no detected lagged linear dependence through ``max_lag`` and smaller is better.
Its mathematical definition is identical for ``evaluation="oos"`` and
``evaluation="in_sample"``; only the source of the one-step errors differs.

The result also exposes the complete matrices ``autocorrelation_matrices``, the
largest absolute lagged correlation, and a multivariate Ljung--Box-style
``portmanteau_statistic``. These summarize closely related residual temporal
structure and should not be interpreted as independent evidence. No p-value is
required for comparing VAR and state-space fits, avoiding model-family-specific
degrees-of-freedom corrections.

The per-variable Durbin--Watson statistic is retained as a supplementary lag-one
diagnostic. It is computed without linking the end of one trajectory to the
beginning of the next.

Consistency and supplementary diagnostics
-----------------------------------------

``consistency`` extends the Ding--Bressler covariance-structure statistic to the
aligned held-out one-step predictions. Larger values indicate that more of the
held-out contemporaneous covariance structure is reproduced by the predictor.
The historical MVGC ``0.8`` rule is a VAR heuristic and is not used as a default
acceptance threshold for state-space models.

State-space recovery when a reference model is known
----------------------------------------------------

Fit diagnostics answer how well a fitted model describes observations. In
simulation studies and estimator benchmarks a different question is available:
how well does an estimated state-space model recover a known reference system?

Raw entry-wise errors in state-space matrices are generally not meaningful. For
an invertible latent coordinate change :math:`T`,

.. math::

   A' = TAT^{-1},\qquad
   C' = CT^{-1},\qquad
   Q' = TQT^\top,

representations related this way describe the same latent dynamics in different
coordinates. :func:`~complextorch.state_space_recovery_diagnostics` therefore
compares quantities that are invariant to this latent-basis freedom.

``spectral_distance`` compares the unordered transition eigenvalue sets using a
symmetric nearest-neighbour distance. It captures recovery of characteristic
state-space timescales and stability structure without imposing an arbitrary
eigenvalue ordering.

``hankel_relative_error`` compares finite innovations Hankel matrices assembled
from Markov blocks. For an innovations model

.. math::

   x_{t+1}=Ax_t+K\varepsilon_t,\qquad
   y_t=Cx_t+\varepsilon_t,\qquad
   \varepsilon_t\sim\mathcal N(0,V),

with :math:`V=LL^\top`, the blocks are :math:`CA^kKL`. Under a latent similarity
transform the factors change but their product does not, making this comparison
gauge-aware. ``hankel_spectrum_relative_error`` compares the corresponding
Hankel singular-value spectra, and ``innovation_covariance_relative_error``
compares :math:`V` directly.

These recovery diagnostics require a known reference model. They are intended
for simulations, benchmarks, and controlled perturbation studies; they do not
provide ground-truth recovery claims for empirical data where the generating
system is unknown.

Minimal workflow
----------------

The following example is intentionally small. It generates a simple stable
multivariate process and shows the recommended reading of fit diagnostics
without turning the User Guide into a benchmarking script.

.. code-block:: python

   import torch
   from complextorch import N4SID, fit_diagnostics

   torch.manual_seed(7)
   x = torch.zeros(1000, 3, dtype=torch.float64)
   a = torch.tensor(
       [[0.65, 0.10, 0.00], [-0.05, 0.55, 0.08], [0.00, 0.04, 0.45]],
       dtype=torch.float64,
   )
   noise = 0.3 * torch.randn_like(x)
   for t in range(1, x.shape[0]):
       x[t] = a @ x[t - 1] + noise[t]

   train, test = x[:800], x[800:]
   model = N4SID(2, block_rows=6, dtype="float64").fit(train)
   diagnostics = fit_diagnostics(model, train, test, max_lag=8)

   # Predictive accuracy
   print(diagnostics.nmse, diagnostics.predictive_r2)

   # Temporal residual adequacy
   print(diagnostics.whiteness_energy)

   # Probabilistic calibration
   print(diagnostics.covariance_calibration)
   print(diagnostics.prediction_interval_coverage)

When a synthetic reference system is available, recovery is a separate check:

.. code-block:: python

   from complextorch import state_space_recovery_diagnostics

   recovery = state_space_recovery_diagnostics(model.system_, reference_system)
   print(recovery.hankel_relative_error)
   print(recovery.spectral_distance)

The second snippet assumes that ``reference_system`` is a known synthetic
:class:`~complextorch.StateSpaceModel` or
:class:`~complextorch.InnovationsStateSpace`.

Batch semantics
---------------

``mode="pooled"`` aggregates sufficient statistics across independent
trajectories while preserving every trajectory boundary. No lagged product ever
crosses a boundary. ``mode="independent"`` returns one scalar diagnostic per
trajectory and preserves batch axes for covariance and lag-correlation matrices.
Prediction means and standardized errors retain the original trajectory layout.

References
----------

- Ding, M., Bressler, S. L., Yang, W., and Liang, H. (2000). *Biological
  Cybernetics*, 83, 35--45.
- Hosking, J. R. M. (1980). *Journal of the American Statistical Association*,
  75, 602--608.
- Kailath, T. (1980). *Linear Systems*. Prentice-Hall.
- Ljung, G. M. and Box, G. E. P. (1978). *Biometrika*, 65, 297--303.
