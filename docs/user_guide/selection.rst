Model and order selection
=========================

ComplexTorch separates fixed-order fitting from model-order selection.
:class:`~complextorch.VAROrderSelectionIC` and
:class:`~complextorch.VAROrderSearchCV` select VAR lag order :math:`p`, whereas
:class:`~complextorch.StateSpaceOrderSelection` and
:class:`~complextorch.StateSpaceOrderSearchCV` select state-space latent
dimension :math:`r`. These are different hyperparameters and must not be
conflated.

VAR information criteria
------------------------

For candidate lag :math:`p`, let :math:`\ell_p` be the maximized Gaussian
log-likelihood contribution per effective observation, :math:`k_p` the number
of model parameters entering the criterion, and :math:`N_p` the effective
sample count. :class:`~complextorch.VAROrderSelectionIC` uses the
MVGC-compatible per-observation forms

.. math::

   \mathrm{AIC}(p)=-2\ell_p+2\frac{k_p}{N_p},

.. math::

   \mathrm{BIC}(p)=-2\ell_p+\frac{k_p}{N_p}\log N_p,

.. math::

   \mathrm{HQC}(p)=-2\ell_p+2\frac{k_p}{N_p}\log\log N_p.

For an :math:`n`-variable VAR,

.. math::

   k_p=pn^2

in the current criterion implementation. For :math:`B` equal-length independent
trajectories of length :math:`T`, ``mode="pooled"`` estimates one common VAR
and uses

.. math::

   N_p=B(T-p).

With ``mode="independent"``, each trajectory is a separate estimation problem,
so each criterion curve uses

.. math::

   N_p=T-p.

For batched input ``(B, T, n)``, independent mode returns ``aic_``, ``bic_``,
``hqc_`` and ``loglik_`` with shape ``(B, n_orders)`` and one selected order per
trajectory with shape ``(B,)``. Candidate VAR fits preserve the batch axis and
never create lagged transitions across trajectories. Because independently
selected trajectories can have different lag orders, batched independent
selection requires ``refit=None``; fitting the selected heterogeneous models is
a separate step.

The optional Hurvich--Tsai correction multiplies the AIC penalty by

.. math::

   \frac{N_p}{N_p-k_p-1}.

Each candidate lag is fitted independently with :class:`~complextorch.VAR`.
The selection procedure therefore does not infer all lower orders from a single
maximal-order fit. The resulting curves are summarized by
:class:`~complextorch.VARInformationCriteriaResult`.

Temporal cross-validation
-------------------------

Time-series CV uses expanding chronological windows rather than shuffled folds.
:class:`~complextorch.EpochTimeSeriesSplit` defines the folds consumed by
:class:`~complextorch.VAROrderSearchCV` and
:class:`~complextorch.StateSpaceOrderSearchCV`. A fold is described by a
training prefix

.. math::

   [0,t_{\mathrm{train}})

and a scored test block

.. math::

   [t_{\mathrm{test}},t_{\mathrm{stop}}),

with optional gap

.. math::

   t_{\mathrm{test}}=t_{\mathrm{train}}+g.

For candidate order :math:`q` and fold loss :math:`L_{qj}`, the finite successful
folds are aggregated as

.. math::

   \bar L_q=\frac1{J_q}\sum_{j\in\mathcal F_q}L_{qj},

with standard error

.. math::

   \operatorname{SE}_q=\frac{s_q}{\sqrt{J_q}}.

With ``selection_rule="best"`` the selected order is

.. math::

   q^*=\arg\min_q\bar L_q.

With ``selection_rule="one_se"``, let

.. math::

   q_{\min}=\arg\min_q\bar L_q.

The selected model is the smallest candidate satisfying

.. math::

   \bar L_q\le
   \bar L_{q_{\min}}+\operatorname{SE}_{q_{\min}}.

This intentionally favors a simpler candidate whose predictive loss is within
one estimated standard error of the minimum. VAR fold-level diagnostics are
represented by :class:`~complextorch.VAROrderScore` and
:class:`~complextorch.VAROrderSearchResult`; the state-space counterparts are
:class:`~complextorch.StateSpaceOrderScore` and
:class:`~complextorch.StateSpaceOrderSearchResult`.

Prediction and gap semantics
----------------------------

``prediction_mode="rolling"`` scores each held-out sample and then allows that
observed sample into the forecasting history. ``prediction_mode="recursive"``
never consumes held-out observations and propagates model predictions
recursively.

``gap_mode="warmup"`` permits unscored gap observations to initialize or update
the predictor before the scored test block. ``gap_mode="embargo"`` excludes gap
observations and propagates only model predictions through the gap. These modes
answer different scientific validation questions and should be reported.

State-space latent dimension
----------------------------

:class:`~complextorch.StateSpaceOrderSelection` selects latent dimension
:math:`r`, not VAR lag order :math:`p`. The selector uses the Larimore canonical
correlations produced by the same subspace machinery used by
:class:`~complextorch.LarimoreStateSpace`:

.. math::

   \rho_1\ge\rho_2\ge\cdots\ge\rho_{r_{\max}}.

For candidate dimension :math:`r`, Bauer's singular-value criterion is

.. math::

   \operatorname{SVC}(r)
   =\rho_{r+1}^2
   +\frac{2n_y r\log N_{\mathrm{eff}}}{N_{\mathrm{eff}}},

where :math:`n_y` is the observation dimension and :math:`N_{\mathrm{eff}}`
the number of valid Hankel columns. The omitted correlation for the full
identifiable rank is defined as zero. Selection is

.. math::

   r^*=\arg\min_r\operatorname{SVC}(r).

The raw canonical correlations are used in Bauer SVC. A normalized spectrum may
be exposed for plotting but does not enter the criterion. Results are summarized
by :class:`~complextorch.StateSpaceOrderSelectionResult`.

For pooled independent trajectories,

.. math::

   N_{\mathrm{eff}}=\sum_b N_{\mathrm{columns}}^{(b)},

where every Hankel column was built inside one trajectory.

State-space temporal CV
-----------------------

:class:`~complextorch.StateSpaceOrderSearchCV` treats latent dimension as a
temporal-validation hyperparameter. Each training fold computes one Larimore
decomposition at the maximum identifiable rank; candidate dimensions truncate
this shared basis and are evaluated only by held-out predictive loss. Bauer SVC
remains a training-only diagnostic rather than the selection objective in this
mode.

References
----------

- Akaike, H. Information-theoretic model selection.
- Schwarz, G. Bayesian information criterion.
- Hannan, E. J. and Quinn, B. G. Hannan--Quinn criterion.
- Bauer, D. (2001). Order estimation for subspace methods. *Automatica*,
  37(10), 1561--1573.
- Larimore, W. E. (1990, 1996). Canonical variate analysis system
  identification.
- Barnett, L. and Seth, A. K. (2014). MVGC toolbox model-order conventions.

Repository references
---------------------

- ``src/complextorch/selection/selection_var.py``
- ``src/complextorch/selection/selection_state_space.py``
- ``src/complextorch/selection/_temporal.py``
- ``src/complextorch/_subspace.py``
