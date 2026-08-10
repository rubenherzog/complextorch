Estimator-agnostic state-space temporal CV
===========================================

:class:`~complextorch.StateSpaceOrderSearchCV` separates leakage-safe temporal
cross-validation from the fixed-order state-space estimator used at each
candidate latent dimension :math:`r`.

``method="larimore"`` preserves the optimized Larimore path: one maximum-rank
CVA decomposition is computed for each training fold and all candidate
state dimensions reuse that basis. ``method="n4sid"`` instead fits
:class:`~complextorch.N4SID` at each candidate dimension under the same folds,
prediction modes, held-out loss, and selection rule. The shorthand maps
``past_horizon`` to the N4SID ``block_rows`` setting.

A scikit-learn-compatible fixed-order estimator can be supplied through
``estimator=``. The prototype is cloned for each candidate and must expose an
``n_states`` parameter. ComplexTorch changes only ``n_states`` and, when the
prototype supports it, ``mode``; other estimator settings remain unchanged.

General state-space estimates such as N4SID are converted to steady-state
innovations form before validation. Consequently Larimore and N4SID are scored
through the same rolling or recursive prediction equations and the same RMSE or
Gaussian negative-log-likelihood objective.

Batch semantics
---------------

Batch semantics match temporal VAR order search. With ``mode="pooled"``, each
batch element is an independent realization of one common system and the search
returns one common latent dimension. No Hankel block, state transition, or
validation prediction crosses a trajectory boundary.

With batched ``mode="independent"``, every batch element is treated as a
different system. Fixed-order candidates are fitted with the estimator's
batched Torch path, while validation scores retain shape
``(batch, n_orders, n_folds)`` and the selected dimensions have shape
``(batch,)``. If the final dimensions differ, ``refit=True`` returns one
fixed-order estimator per trajectory because heterogeneous latent dimensions
cannot be represented by one dense state-space tensor.

Diagnostics and validation
--------------------------

Bauer SVC is specific to the Larimore canonical-correlation decomposition and
therefore remains a Larimore-only training diagnostic. It does not participate
in temporal-CV selection. N4SID and custom estimators expose held-out temporal
scores without fabricating a Bauer curve.

Synthetic regression tests cover a known two-state Gaussian system. They check
that N4SID temporal CV prefers the correct two-dimensional model to a one-state
underfit candidate, that the refitted observation subspace recovers the true
latent observation subspace, and that batched ``mode="independent"`` is
numerically equivalent to running the public search separately on every
trajectory.

References
----------

- Van Overschee, P. and De Moor, B. (1994). N4SID: Subspace algorithms for the
  identification of combined deterministic-stochastic systems. *Automatica*,
  30(1), 75--93.
- Larimore, W. E. (1990, 1996). Canonical variate analysis for system
  identification.
- Bauer, D. (2001). Order estimation for subspace methods. *Automatica*,
  37(10), 1561--1573.
