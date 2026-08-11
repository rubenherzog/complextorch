Fit diagnostics
===============

ComplexTorch separates predictive accuracy from innovations-model adequacy.
:func:`~complextorch.fit_diagnostics` supports two explicit evaluation regimes
for a fitted :class:`~complextorch.VAR`, :class:`~complextorch.LarimoreStateSpace`,
or :class:`~complextorch.N4SID`: ``evaluation="oos"`` evaluates a chronologically
later test block using rolling one-step-ahead prediction, whereas
``evaluation="in_sample"`` evaluates one-step residuals on the supplied fitted-data
trajectory. The estimator is never refitted by the diagnostic. Both regimes share
the same lower-level :func:`~complextorch.innovation_diagnostics` implementation.

For observations :math:`y_t` and one-step predictions
:math:`\hat y_{t\mid t-1}`, define

.. math::

   e_t = y_t-\hat y_{t\mid t-1}.

Prediction quality
------------------

The result reports RMSE, Gaussian negative log likelihood, normalized mean
squared error,

.. math::

   \operatorname{NMSE}
   = \frac{\sum_t\lVert e_t\rVert_2^2}
           {\sum_t\lVert y_t-\bar y_{\mathrm{train}}\rVert_2^2},

and the corresponding predictive coefficient
:math:`R^2_{\mathrm{pred}}=1-\operatorname{NMSE}`. These quantities answer how
well the fitted model predicts unseen observations.

Innovation calibration
----------------------

Let :math:`V` denote the innovation covariance estimated on the training data.
The covariance of the evaluated one-step innovations is reported directly
(``innovation_covariance_oos`` retains its original public field name for backward
compatibility). Errors are standardized by a Cholesky factor
:math:`V=LL^\top`,

.. math::

   z_t=L^{-1}e_t.

A calibrated innovations model should have approximately zero standardized mean
and identity standardized covariance. ``covariance_calibration`` is

.. math::

   \frac{\lVert\widehat\Sigma_z-I\rVert_F}{\sqrt n},

so smaller values indicate better covariance calibration.

Multivariate whiteness
----------------------

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
``portmanteau_statistic``. No p-value is required for comparing VAR and
state-space fits, avoiding model-family-specific degrees-of-freedom corrections.

The per-variable Durbin--Watson statistic is retained as a supplementary lag-one
diagnostic. It is computed without linking the end of one trajectory to the
beginning of the next.

Consistency
-----------

``consistency`` extends the Ding--Bressler covariance-structure statistic to the
aligned held-out one-step predictions. Larger values indicate that more of the
held-out contemporaneous covariance structure is reproduced by the predictor.
The historical MVGC ``0.8`` rule is a VAR heuristic and is not used as a default
acceptance threshold for state-space models.

Batch semantics
---------------

``mode="pooled"`` aggregates sufficient statistics across independent
trajectories while preserving every trajectory boundary. No lagged product ever
crosses a boundary. ``mode="independent"`` returns one scalar diagnostic per
trajectory and preserves batch axes for covariance and lag-correlation matrices.

References
----------

- Ding, M., Bressler, S. L., Yang, W., and Liang, H. (2000). *Biological
  Cybernetics*, 83, 35--45.
- Hosking, J. R. M. (1980). *Journal of the American Statistical Association*,
  75, 602--608.
- Ljung, G. M. and Box, G. E. P. (1978). *Biometrika*, 65, 297--303.
