State-space fitting and optional EM refinement
==============================================

ComplexTorch keeps state-space identification and likelihood refinement as
separate steps. Subspace estimators such as :class:`~complextorch.N4SID` provide
fast fixed-order identification, while :class:`~complextorch.LinearGaussianEM`
can optionally refine a fitted general linear-Gaussian state-space model by
expectation--maximization.

A fitted N4SID estimator can be passed directly to the existing EM estimator:

.. code-block:: python

   from complextorch import LinearGaussianEM, N4SID

   initial = N4SID(4, block_rows=8, dtype="float64").fit(data)
   refined = LinearGaussianEM(initial, n_iter=20).fit(data)

   print(refined.initial_log_likelihood_)
   print(refined.final_log_likelihood_)
   print(refined.log_likelihood_gain_)

No second refinement API is required. When initialized from a fitted N4SID
estimator, EM reuses the fitted centering convention and exposes both aggregate
and per-trajectory likelihood changes. ``mode="pooled"`` estimates one common
model across independent trajectories while preserving their boundaries;
``mode="independent"`` refines one model per trajectory.

EM is optional
--------------

EM is not part of model-order selection and does not replace the subspace fit.
It is useful when likelihood refinement of a chosen fixed-order general
state-space model is desired. The appropriate workflow is therefore

``select order -> fit subspace model -> optionally refine -> diagnose``.

The refinement should be evaluated using the same temporal validation design as
the unrefined model. An increased training likelihood alone is not evidence of
better out-of-sample prediction or better residual adequacy; use
:func:`~complextorch.fit_diagnostics` for those questions.

Innovations-form limitation
---------------------------

The current :class:`~complextorch.LinearGaussianEM` contract assumes independent
process and observation noise in a general state-space model. An innovations
representation

.. math::

   x_{t+1}=Ax_t+K\varepsilon_t,\qquad
   y_t=Cx_t+\varepsilon_t

uses the same innovation in both equations and therefore corresponds to
correlated process and observation noise when rewritten in general form.
Consequently, fitted :class:`~complextorch.LarimoreStateSpace` innovations-form
models are not silently converted and refined by this EM implementation. Exact
support would require an EM algorithm with the corresponding noise
cross-covariance rather than an approximate conversion.
