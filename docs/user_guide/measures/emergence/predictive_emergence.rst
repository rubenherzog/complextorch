Rosas--Mediano causal emergence
===============================

ComplexTorch implements the practical finite-delay criteria from Rosas,
Mediano et al. (2020), *Reconciling emergences*. For a user-specified linear
macro-feature :math:`V_t=LX_t` and a positive delay :math:`\tau`, the published
order-one criteria are

.. math::

   \Psi_\tau(V)
   = I(V_t;V_{t+\tau})
   - \sum_j I(X_t^j;V_{t+\tau}),

.. math::

   \Delta_\tau(V)
   = \max_j\left[
   I(V_t;X_{t+\tau}^j)
   - \sum_i I(X_t^i;X_{t+\tau}^j)\right],

.. math::

   \Gamma_\tau(V)
   = \max_j I(V_t;X_{t+\tau}^j).

Use :func:`~complextorch.measures.emergence_from_model` for the primary
model-based calculation. For stationary Gaussian ``VARSystem`` and
``InnovationsStateSpace`` models, ComplexTorch evaluates the mutual
informations exactly from :math:`\Gamma_0` and :math:`\Gamma_\tau`; no time
series are simulated and no PCA or other macro-feature selection is performed.
``lag=1`` is the default, and ``lag`` may be any positive integer supported by
the model autocovariance backbone.

The implementation is pinned for parity to the reference MATLAB functions
``EmergencePsi.m``, ``EmergenceDelta.m``, and ``EmergenceGamma.m`` in
``pmediano/ReconcilingEmergences`` commit
``ecf591aacb6d58996c903b51a2f945cd7f713a32``.

Distinction from dynamical dependence
-------------------------------------

These quantities are not SSDI dynamical dependence. In particular,
:math:`\Psi_\tau` is a whole-minus-sum criterion built from finite-delay mutual
information. :func:`~complextorch.dynamical_dependence` is the Barnett--Seth
reduced-innovations / Granger-causality quantity. ComplexTorch keeps the two
families separate even when the same projection :math:`L` is supplied.

Full-past extension
-------------------

``history="full"`` is reserved for a possible ComplexTorch extension and is
not part of Rosas et al. (2020). The proposed quantities involving
:math:`V_{<t}` or :math:`X^i_{<t}` are mathematically computable for linear
Gaussian state-space models by generalized-DARE prediction conditioned on each
projected history. They require conditional prediction covariances for arbitrary
projected histories, not only reduced innovation covariances. ComplexTorch does
not yet expose that reusable primitive, so the extension is intentionally not
implemented rather than duplicating the control/DARE core.

References
----------

- Rosas, F. E., Mediano, P. A. M., Jensen, H. J., Seth, A. K., Barrett, A. B.,
  Carhart-Harris, R. L., and Bor, D. (2020). *Reconciling emergences: An
  information-theoretic approach to identify causal emergence in multivariate
  data*. PLoS Computational Biology 16(12): e1008289.
- Barnett, L. and Seth, A. K. (2023). *Dynamical independence and emergent
  macroscopic processes*. Physical Review E 108, 014304.
