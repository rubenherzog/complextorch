Rosas--Mediano emergence
========================

ComplexTorch implements the practical causal-emergence criteria introduced by
Rosas, Mediano *et al.* (2020) for a user-specified macro-variable
:math:`V_t=LX_t`.  ComplexTorch does **not** estimate ``L`` inside the measure;
the macro-projection is part of the scientific question and must be supplied by
the caller.

Published finite-delay criteria
--------------------------------

For a positive delay :math:`\tau`, the published practical criteria are

.. math::

   \Psi_\tau(V)
   = I(V_t;V_{t+\tau})
   - \sum_j I(X_t^j;V_{t+\tau}),

.. math::

   \Delta_\tau(V)
   = \max_j\left[
   I(V_t;X_{t+\tau}^j)
   - \sum_i I(X_t^i;X_{t+\tau}^j)
   \right],

.. math::

   \Gamma_\tau(V)
   = \max_j I(V_t;X_{t+\tau}^j).

These definitions follow Rosas *et al.* and the accompanying reference MATLAB
implementations ``EmergencePsi.m``, ``EmergenceDelta.m`` and
``EmergenceGamma.m`` in ``pmediano/ReconcilingEmergences`` at commit
``ecf591aacb6d58996c903b51a2f945cd7f713a32``.  ``lag`` is :math:`\tau` and
defaults to 1.

For stationary Gaussian models, ComplexTorch evaluates all mutual informations
analytically from :math:`\Gamma_0` and :math:`\Gamma_\tau`.  No trajectory is
simulated and no covariance is re-estimated from observations.  The model-first
entry point is
:func:`~complextorch.measures.emergence_from_model`; the observation plug-in
estimator :func:`~complextorch.measures.emergence_from_observations` implements
only these published finite-delay quantities.

ComplexTorch full-past extension
--------------------------------

ComplexTorch additionally defines the optional ``history="full"`` extension

.. math::

   \Psi_\infty(V)
   = I(V_{<t};V_t)
   - \sum_j I(X^j_{<t};V_t),

.. math::

   \Delta_\infty(V)
   = \max_j\left[
   I(V_{<t};X_t^j)
   - \sum_i I(X^i_{<t};X_t^j)
   \right],

.. math::

   \Gamma_\infty(V)
   = \max_j I(V_{<t};X_t^j).

This full-past variant is a **ComplexTorch extension** and must not be
attributed to Rosas *et al.*  It replaces each finite delayed source by its
complete semi-infinite past.  For linear-Gaussian models it is evaluated
exactly from the canonical innovations representation using the same projected
generalized-DARE machinery used elsewhere in ComplexTorch.  In particular,
if :math:`Y=LX` is the conditioning process and :math:`P_L` is the steady-state
state prediction covariance obtained from the projected-history DARE, then

.. math::

   \operatorname{Cov}(X_t\mid Y_{<t})
   = C P_L C^\top + \Sigma.

The implementation batches the singleton microscopic histories through this
shared Riccati primitive; it does not truncate the past or simulate data.
``lag`` has no role when ``history="full"``.

API
---

.. code-block:: python

   result = emergence_from_model(
       model,
       macro_projection,
       lag=1,
       history="lagged",  # "lagged" or "full"
       base=2.0,
   )

The result contains ``psi``, ``delta`` and ``gamma`` together with the mutual
information terms from which they are assembled.

Relation to dynamical dependence
--------------------------------

Rosas--Mediano :math:`\Psi`, :math:`\Delta` and :math:`\Gamma` are distinct
from Barnett--Seth dynamical dependence.  Dynamical dependence is a projected
Granger-causality / transfer-entropy quantity based on reduced innovations;
it must not be used as an alternative definition of :math:`\Psi`.

References
----------

- Rosas, F. E., Mediano, P. A. M., Jensen, H. J., Seth, A. K., Barrett, A. B.,
  Carhart-Harris, R. L. and Bor, D. (2020). Reconciling emergences: An
  information-theoretic approach to identify causal emergence in multivariate
  data. *PLoS Computational Biology*, 16(12), e1008289.
- Barnett, L. and Seth, A. K. (2023). Dynamical independence: Discovering
  emergent macroscopic processes in complex dynamical systems. *Physical
  Review E*, 108, 014304.
