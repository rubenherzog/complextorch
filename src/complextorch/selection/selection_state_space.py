r"""State-space latent-dimension selection.

This module contains the state-space selectors, while reusable Larimore CVA
linear-algebra primitives remain in :mod:`complextorch._subspace` and fixed-
dimension fitting remains in :mod:`complextorch.state_space`.

``StateSpaceOrderSelection`` applies Bauer SVC to a Larimore canonical-
correlation spectrum. ``StateSpaceOrderSearchCV`` instead treats latent
dimension as a temporal-validation hyperparameter: each fold computes one
Larimore decomposition at maximum identifiable rank, candidate dimensions
truncate the shared basis, and only held-out predictive loss selects the final
dimension. Bauer SVC is exposed there as a training-only diagnostic.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Iterable, Literal

import numpy as np
import torch
from sklearn.base import BaseEstimator

from .._subspace import _block_hankel, _larimore_decomposition
from ..control import InnovationsStateSpace
from ..linalg import stable_cholesky
from ..state_space import (
    LarimoreStateSpace,
    _fit_innovations_state_space_from_states,
    _normalise_ss_observations,
)
from ._temporal import EpochTimeSeriesSplit, _TemporalOrderSearchCV


def _bauer_svc(
    canonical_correlations: np.ndarray | torch.Tensor,
    n_observations: int,
    n_effective: int | torch.Tensor,
    *,
    min_order: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Select state dimension with Bauer's singular-value criterion.

    Parameters
    ----------
    canonical_correlations
        Unnormalised canonical correlations in descending order with shape
        ``(..., r_max)``. Leading dimensions are processed independently.
    n_observations
        Number :math:`n_y` of observed variables.
    n_effective
        Effective number :math:`N_{\mathrm{eff}}` of Hankel columns. A scalar
        or tensor broadcastable over the leading batch dimensions.
    min_order
        Smallest candidate state dimension. Defaults to one. Latent state
        dimension is not constrained by the number of observed variables.

    Returns
    -------
    best_order
        Tensor containing the minimizing state dimension for each leading
        batch element.
    criterion
        SVC values with shape ``(..., r_max - min_order + 1)``.

    Notes
    -----
    For candidate dimension :math:`r`, Bauer SVC is

    .. math::

       \operatorname{SVC}(r)
       = \rho_{r+1}^{2}
         + \frac{2 n_y r\log N_{\mathrm{eff}}}{N_{\mathrm{eff}}}.

    The canonical correlations are clipped to ``[0, 1]`` only to remove
    floating-point excursions. They are never normalized by the leading
    correlation before applying SVC.

    References
    ----------
    - Bauer, D. (2001). Order estimation for subspace methods. *Automatica*,
      37(10), 1561--1573.
    - ComplexBox ``mvgc.modelorder.bauer_svc``.
    """
    rho = torch.as_tensor(canonical_correlations)
    if rho.ndim < 1 or rho.shape[-1] == 0:
        raise ValueError("canonical_correlations must end in a non-empty axis")
    if not rho.is_floating_point():
        rho = rho.to(torch.float64)
    if not torch.isfinite(rho).all():
        raise ValueError("canonical_correlations must be finite")
    if n_observations < 1:
        raise ValueError("n_observations must be positive")

    r_max = rho.shape[-1]
    lower = 1 if min_order is None else int(min_order)
    if lower < 1:
        raise ValueError("min_order must be at least one")
    if lower > r_max:
        raise ValueError(
            f"minimum order {lower} exceeds maximum identifiable order {r_max}"
        )

    effective = torch.as_tensor(
        n_effective, dtype=rho.dtype, device=rho.device
    )
    if torch.any(effective <= 1) or not torch.isfinite(effective).all():
        raise ValueError("n_effective must be finite and greater than one")

    orders = torch.arange(lower, r_max + 1, device=rho.device)
    # Append the theoretical zero omitted correlation for the full-rank model;
    # zero-based index r then retrieves rho_{r+1} in one-based notation.
    padded = torch.cat(
        (rho.clamp(0.0, 1.0), torch.zeros_like(rho[..., :1])), dim=-1
    )
    omitted = padded.index_select(-1, orders)

    while effective.ndim < rho.ndim - 1:
        effective = effective.unsqueeze(-1)
    penalty = (
        2.0
        * float(n_observations)
        * orders.to(rho.dtype)
        * torch.log(effective.unsqueeze(-1))
        / effective.unsqueeze(-1)
    )
    criterion = omitted.square() + penalty
    best_order = orders[criterion.argmin(dim=-1)]
    return best_order, criterion


@dataclass(frozen=True)
class _StateSpaceOrderComputation:
    """Result of Larimore CVA followed by Bauer SVC."""

    best_order: torch.Tensor
    candidate_orders: torch.Tensor
    criterion: torch.Tensor
    canonical_correlations: torch.Tensor
    normalized_canonical_correlations: torch.Tensor
    n_effective: torch.Tensor


def _larimore_state_space_order(
    observations: np.ndarray | torch.Tensor,
    past_horizon: int,
    *,
    future_horizon: int | None = None,
    min_order: int | None = None,
    mode: Literal["pooled", "independent"] = "pooled",
    ridge: float = 1e-12,
    device: str | torch.device = "auto",
    dtype: str | torch.dtype = "float64",
) -> _StateSpaceOrderComputation:
    r"""Estimate latent state order by Larimore CVA and Bauer SVC.

    Parameters
    ----------
    observations
        Time series in ``(time, variables)`` or ComplexTorch batch-first
        ``(batch, time, variables)`` layout.
    past_horizon, future_horizon
        Numbers of block rows in the past and future Hankel matrices. The
        future horizon defaults to the past horizon.
    min_order
        Smallest candidate state dimension. Defaults to one.
    mode
        ``"pooled"`` aggregates valid Hankel columns across independent
        trajectories; ``"independent"`` computes one order per trajectory.
        Trajectory boundaries are never connected.
    ridge
        Non-negative diagonal regularizer used only for Cholesky whitening.
    device, dtype
        Torch execution device and floating-point dtype.

    Returns
    -------
    _StateSpaceOrderComputation
        Canonical correlations, SVC curve, candidate dimensions, effective
        sample size, and selected state order.

    References
    ----------
    - Larimore, W. E. (1990, 1996), canonical variate state construction.
    - Bauer, D. (2001), singular-value model-order estimation.
    - ComplexBox ``mvgc.modelorder.tsdata_to_ssmo``.
    """
    if past_horizon < 1:
        raise ValueError("past_horizon must be positive")
    future = past_horizon if future_horizon is None else int(future_horizon)
    if future < 1:
        raise ValueError("future_horizon must be positive")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    if mode not in {"pooled", "independent"}:
        raise ValueError("mode must be 'pooled' or 'independent'")

    values, _ = _normalise_ss_observations(
        observations, device=device, dtype=dtype
    )
    n_batch, n_times, n_variables = values.shape
    if past_horizon + future > n_times:
        raise ValueError("past/future horizons are too large for the series")

    # Center each trajectory independently before pooled sufficient statistics;
    # this preserves independent-trial semantics and never introduces a
    # transition or Hankel column spanning two trajectories.
    values = values - values.mean(dim=1, keepdim=True)
    past, future_blocks = _block_hankel(values, past_horizon, future)
    columns_per_trajectory = past.shape[-1]

    if mode == "pooled":
        # Concatenate valid Hankel columns, not raw time series. Each trajectory
        # therefore contributes statistics without boundary-crossing lags.
        past = past.permute(1, 0, 2).reshape(past.shape[1], -1).unsqueeze(0)
        future_blocks = (
            future_blocks.permute(1, 0, 2)
            .reshape(future_blocks.shape[1], -1)
            .unsqueeze(0)
        )
        n_effective = torch.tensor(
            n_batch * columns_per_trajectory,
            device=values.device,
            dtype=values.dtype,
        )
    else:
        n_effective = torch.full(
            (n_batch,),
            columns_per_trajectory,
            device=values.device,
            dtype=values.dtype,
        )

    correlations, _, _ = _larimore_decomposition(
        past, future_blocks, ridge=ridge
    )
    r_max = correlations.shape[-1]
    lower = 1 if min_order is None else int(min_order)
    best_order, criterion = _bauer_svc(
        correlations,
        n_observations=n_variables,
        n_effective=n_effective,
        min_order=lower,
    )
    candidate_orders = torch.arange(lower, r_max + 1, device=values.device)

    # Normalization is visualization-only; Bauer always receives the raw
    # canonical correlations above.
    leading = correlations[..., :1]
    normalized = torch.where(
        leading > 0,
        correlations / leading,
        correlations,
    )
    if mode == "pooled":
        best_order = best_order.squeeze(0)
        criterion = criterion.squeeze(0)
        correlations = correlations.squeeze(0)
        normalized = normalized.squeeze(0)

    return _StateSpaceOrderComputation(
        best_order=best_order,
        candidate_orders=candidate_orders,
        criterion=criterion,
        canonical_correlations=correlations,
        normalized_canonical_correlations=normalized,
        n_effective=n_effective,
    )


@dataclass(frozen=True)
class StateSpaceOrderSelectionResult:
    """Immutable result of state-space latent-order selection."""

    best_order: torch.Tensor
    orders: torch.Tensor
    criterion: torch.Tensor
    criterion_name: str
    subspace_method: str
    canonical_correlations: torch.Tensor
    normalized_canonical_correlations: torch.Tensor
    n_effective: torch.Tensor


class StateSpaceOrderSelection(BaseEstimator):
    r"""Select latent state dimension using Larimore CVA and Bauer SVC.

    This estimator selects latent state dimension :math:`r`, not VAR lag order
    :math:`p`. Larimore CVA provides the unnormalised canonical-correlation
    spectrum and Bauer SVC chooses the minimizing candidate dimension.

    Parameters
    ----------
    past_horizon
        Number of block rows in the past Hankel matrix.
    future_horizon
        Number of block rows in the future Hankel matrix. Defaults to
        ``past_horizon``.
    min_order
        Smallest candidate state dimension. Defaults to one; latent dimension
        is not constrained by the observation dimension.
    subspace_method
        Subspace spectrum estimator. Only ``"larimore"`` is implemented.
    criterion
        State-order criterion. Only ``"bauer"`` is implemented.
    mode
        ``"pooled"`` aggregates Hankel statistics across trajectories;
        ``"independent"`` selects one order per trajectory.
    ridge
        Non-negative diagonal regularizer used only in Cholesky whitening.
    device, dtype
        Torch execution device and floating-point dtype.
    refit
        If ``True``, fit :class:`LarimoreStateSpace` at the selected dimension.
        This is currently supported only in pooled mode because independent
        trajectories may select different dimensions.

    References
    ----------
    - Larimore, W. E. (1990, 1996), canonical variate analysis for system
      identification.
    - Bauer, D. (2001). Order estimation for subspace methods. *Automatica*,
      37(10), 1561--1573.
    """

    def __init__(
        self,
        past_horizon: int,
        *,
        future_horizon: int | None = None,
        min_order: int | None = None,
        subspace_method: Literal["larimore"] = "larimore",
        criterion: Literal["bauer"] = "bauer",
        mode: Literal["pooled", "independent"] = "pooled",
        ridge: float = 1e-12,
        device: str | torch.device = "auto",
        dtype: str | torch.dtype = "float64",
        refit: bool = False,
    ):
        """Initialize state-space order-selection settings."""
        self.past_horizon = past_horizon
        self.future_horizon = future_horizon
        self.min_order = min_order
        self.subspace_method = subspace_method
        self.criterion = criterion
        self.mode = mode
        self.ridge = ridge
        self.device = device
        self.dtype = dtype
        self.refit = refit

    def fit(self, X: np.ndarray | torch.Tensor, y=None):
        """Estimate the canonical spectrum and select latent state order."""
        del y
        if self.subspace_method != "larimore":
            raise ValueError("subspace_method must be 'larimore'")
        if self.criterion != "bauer":
            raise ValueError("criterion must be 'bauer'")
        if self.refit and self.mode != "pooled":
            raise ValueError(
                "refit=True currently requires mode='pooled'; independent "
                "trajectories may select different state dimensions"
            )

        computation = _larimore_state_space_order(
            X,
            self.past_horizon,
            future_horizon=self.future_horizon,
            min_order=self.min_order,
            mode=self.mode,
            ridge=self.ridge,
            device=self.device,
            dtype=self.dtype,
        )
        result = StateSpaceOrderSelectionResult(
            best_order=computation.best_order,
            orders=computation.candidate_orders,
            criterion=computation.criterion,
            criterion_name=self.criterion,
            subspace_method=self.subspace_method,
            canonical_correlations=computation.canonical_correlations,
            normalized_canonical_correlations=(
                computation.normalized_canonical_correlations
            ),
            n_effective=computation.n_effective,
        )
        self.result_ = result
        self.best_order_ = result.best_order
        self.orders_ = result.orders
        self.criterion_ = result.criterion
        self.criterion_name_ = result.criterion_name
        self.subspace_method_ = result.subspace_method
        self.canonical_correlations_ = result.canonical_correlations
        self.normalized_canonical_correlations_ = (
            result.normalized_canonical_correlations
        )
        self.n_effective_ = result.n_effective
        if self.refit:
            self.best_estimator_ = LarimoreStateSpace(
                n_states=int(self.best_order_),
                past_horizon=self.past_horizon,
                future_horizon=self.future_horizon,
                ridge=self.ridge,
                mode="pooled",
                device=self.device,
                dtype=self.dtype,
            ).fit(X)
        return self


@dataclass(frozen=True)
class StateSpaceOrderScore:
    """Held-out and Bauer diagnostic results for one latent dimension.

    Scalar fields remain scalars for a single series or pooled search. For
    batched ``mode="independent"`` they are arrays with one value per
    trajectory; fold-level fields have shape ``(batch, n_folds)``.
    """

    order: int
    mean_score: float | np.ndarray
    standard_error: float | np.ndarray
    fold_scores: tuple[float, ...] | np.ndarray
    failed_folds: int | np.ndarray
    mean_bauer_score: float | np.ndarray
    fold_bauer_scores: tuple[float, ...] | np.ndarray


@dataclass(frozen=True)
class StateSpaceOrderSearchResult:
    """Immutable summary of temporal state-space order search."""

    best_order: int | np.ndarray
    scores: tuple[StateSpaceOrderScore, ...]
    scoring: str
    selection_rule: str

    def as_records(self):
        """Return records suitable for tabular serialization."""
        return [score.__dict__.copy() for score in self.scores]


class _LarimoreFoldWorkspace:
    """One reusable Larimore decomposition for a temporal training fold."""

    def __init__(self, values, *, past_horizon, future_horizon, ridge, mode):
        """Build and cache one fold-level Larimore CVA decomposition."""
        self.values = values
        self.past_horizon = past_horizon
        self.future_horizon = future_horizon
        self.ridge = ridge
        self.mode = mode
        past, future = _block_hankel(values, past_horizon, future_horizon)
        self.batch = values.shape[0]
        self.n_columns = past.shape[-1]
        if mode == "pooled":
            self.past_fit = (
                past.permute(1, 0, 2)
                .reshape(past.shape[1], -1)
                .unsqueeze(0)
            )
            self.future_fit = (
                future.permute(1, 0, 2)
                .reshape(future.shape[1], -1)
                .unsqueeze(0)
            )
        else:
            self.past_fit, self.future_fit = past, future
        (
            self.correlations,
            self.right_vectors,
            self.cholesky_past,
        ) = _larimore_decomposition(
            self.past_fit, self.future_fit, ridge=ridge
        )
        # Solve L_P^T X = P to obtain L_P^{-T} P without an explicit inverse.
        self.whitened_past = torch.linalg.solve_triangular(
            self.cholesky_past.transpose(-1, -2),
            self.past_fit,
            upper=True,
        )

    def fit_order(self, order, *, covariance):
        """Truncate the shared canonical basis and estimate ``A,C,K,V``."""
        if order > self.correlations.shape[-1]:
            raise ValueError("state dimension exceeds identifiable subspace rank")
        # Larimore state coordinates are the retained canonical directions,
        # weighted by their canonical correlations, applied to whitened pasts.
        state_columns = (
            self.correlations[..., :order].unsqueeze(-1)
            * self.right_vectors[..., :order, :]
        ) @ self.whitened_past
        if self.mode == "pooled":
            states = state_columns.squeeze(0).T.reshape(
                self.batch, self.n_columns, order
            )
        else:
            states = state_columns.transpose(-1, -2)
        transition, observation, gain, innovation_covariance, innovations = (
            _fit_innovations_state_space_from_states(
                self.values,
                states,
                observation_start=self.past_horizon,
                mode=self.mode,
                covariance=covariance,
                min_covar=max(self.ridge, 1e-12),
            )
        )
        system = InnovationsStateSpace(
            transition, observation, gain, innovation_covariance
        )
        return SimpleNamespace(
            system_=system,
            transition_=transition,
            observation_=observation,
            kalman_gain_=gain,
            innovation_covariance_=innovation_covariance,
            innovations_=innovations,
            states_=states,
            n_states_=order,
        )


class StateSpaceOrderSearchCV(_TemporalOrderSearchCV):
    r"""Select latent dimension by temporal cross-validation.

    For batched ``mode="independent"`` input, every trajectory retains its
    own validation-loss curve and selected latent dimension. ``mode="pooled"``
    selects one common dimension across trajectories.

    ``prediction_mode='rolling'`` updates the innovations state with each
    validation observation after scoring it. ``'recursive'`` propagates only
    the fitted model during validation. ``gap_mode='warmup'`` consumes unscored
    gap observations, while ``'embargo'`` excludes them and propagates only
    :math:`z_{t+1}=Az_t` across the gap.

    A single Larimore CVA decomposition is computed per fold and shared across
    all candidate dimensions. Bauer SVC remains a training-only diagnostic and
    uses the same candidate state dimensions as the CV search; latent order is
    not lower-bounded by the observation dimension. Innovation covariance uses
    maximum-likelihood normalization by default, matching
    :class:`LarimoreStateSpace`.
    """

    def __init__(
        self,
        orders: Iterable[int],
        past_horizon: int,
        *,
        future_horizon: int | None = None,
        cv: EpochTimeSeriesSplit | None = None,
        scoring: Literal["nll", "rmse"] = "nll",
        selection_rule: Literal["best", "one_se"] = "one_se",
        ridge: float = 1e-12,
        covariance: Literal["mle", "unbiased"] = "mle",
        mode: Literal["pooled", "independent"] = "pooled",
        device: str | torch.device = "auto",
        dtype: str | torch.dtype = "float64",
        prediction_mode: Literal["rolling", "recursive"] = "rolling",
        gap_mode: Literal["warmup", "embargo"] = "warmup",
        refit: bool = True,
        bauer_diagnostics: bool = True,
    ):
        """Initialize temporal state-space order-search settings."""
        self._set_temporal_search_parameters(
            orders=orders,
            cv=cv,
            scoring=scoring,
            selection_rule=selection_rule,
            prediction_mode=prediction_mode,
            gap_mode=gap_mode,
            refit=refit,
        )
        self.past_horizon = past_horizon
        self.future_horizon = future_horizon
        self.ridge = ridge
        self.covariance = covariance
        self.mode = mode
        self.device = device
        self.dtype = dtype
        self.bauer_diagnostics = bauer_diagnostics

    def _validate_common_settings(self):
        """Validate shared and Larimore-specific search settings."""
        orders = super()._validate_common_settings()
        future = (
            self.past_horizon
            if self.future_horizon is None
            else self.future_horizon
        )
        if self.past_horizon < 1 or future < 1:
            raise ValueError("past_horizon and future_horizon must be positive")
        if self.ridge < 0:
            raise ValueError("ridge must be non-negative")
        if self.covariance not in {"mle", "unbiased"}:
            raise ValueError("covariance must be 'mle' or 'unbiased'")
        if self.mode not in {"pooled", "independent"}:
            raise ValueError("mode must be 'pooled' or 'independent'")
        return orders

    def _score_batch_shape(self, data):
        """Preserve one validation curve per independent batch trajectory."""
        if self.mode == "independent" and data.shape[0] > 1:
            return (data.shape[0],)
        return ()

    def _minimum_training_size(self, orders):
        """Return the minimum fold length required by horizons and rank."""
        future = (
            self.past_horizon
            if self.future_horizon is None
            else self.future_horizon
        )
        return self.past_horizon + future + max(orders) + 2

    def _begin_diagnostics(self, n_orders, n_folds):
        """Allocate fold-wise Bauer diagnostic arrays."""
        shape = getattr(self, "_score_batch_shape_", ()) + (n_orders, n_folds)
        self._bauer_scores = np.full(shape, np.nan)
        self._bauer_orders = [None] * n_folds

    def _prepare_fold(self, data, fold, orders):
        """Center one fold and compute its reusable Larimore workspace."""
        values, _ = _normalise_ss_observations(
            data, device=self.device, dtype=self.dtype
        )
        # Use the training mean for the entire fold so held-out observations do
        # not influence preprocessing.
        training_mean = values[:, : fold.train_stop].mean(dim=1, keepdim=True)
        centered = values - training_mean
        future = (
            self.past_horizon
            if self.future_horizon is None
            else self.future_horizon
        )
        workspace = _LarimoreFoldWorkspace(
            centered[:, : fold.train_stop],
            past_horizon=self.past_horizon,
            future_horizon=future,
            ridge=self.ridge,
            mode=self.mode,
        )
        return {
            "decomposition": workspace,
            "centered": centered,
            "orders": orders,
        }

    def _bauer_fold(self, training, orders):
        """Compatibility hook returning a Bauer curve for one training fold."""
        future = (
            self.past_horizon
            if self.future_horizon is None
            else self.future_horizon
        )
        workspace = _LarimoreFoldWorkspace(
            training,
            past_horizon=self.past_horizon,
            future_horizon=future,
            ridge=self.ridge,
            mode=self.mode,
        )
        n_variables = training.shape[-1]
        lower = min(orders)
        curve = np.full(len(orders), np.nan)
        if lower > workspace.correlations.shape[-1]:
            return curve, None
        best, criterion = _bauer_svc(
            workspace.correlations,
            n_variables,
            workspace.n_columns
            * (workspace.batch if self.mode == "pooled" else 1),
            min_order=lower,
        )
        candidates = torch.arange(lower, workspace.correlations.shape[-1] + 1)
        for index, order in enumerate(orders):
            match = torch.nonzero(candidates == order, as_tuple=False)
            if match.numel():
                curve[index] = float(criterion[..., int(match[0])].mean())
        best_values = best.detach().cpu().reshape(-1)
        selected = (
            int(best_values[0])
            if best_values.numel() == 1
            else tuple(int(value) for value in best_values)
        )
        return curve, selected

    def _fold_diagnostics(self, workspace, orders, fold_index):
        """Store Bauer diagnostics without participating in CV selection."""
        workspace["fold_index"] = fold_index
        if not self.bauer_diagnostics:
            return
        if type(self)._bauer_fold is not StateSpaceOrderSearchCV._bauer_fold:
            curve, selected = self._bauer_fold(
                workspace["decomposition"].values, orders
            )
            self._bauer_scores[:, fold_index] = curve
            self._bauer_orders[fold_index] = selected
            return
        decomposition = workspace["decomposition"]
        n_variables = decomposition.values.shape[-1]
        lower = min(orders)
        if lower > decomposition.correlations.shape[-1]:
            return
        best, criterion = _bauer_svc(
            decomposition.correlations,
            n_variables,
            decomposition.n_columns
            * (decomposition.batch if self.mode == "pooled" else 1),
            min_order=lower,
        )
        candidates = torch.arange(
            lower, decomposition.correlations.shape[-1] + 1
        )
        independent_batch = self.mode == "independent" and decomposition.batch > 1
        for index, order in enumerate(orders):
            match = torch.nonzero(candidates == order, as_tuple=False)
            if match.numel():
                values = criterion[..., int(match[0])].detach().cpu().numpy()
                if independent_batch:
                    self._bauer_scores[..., index, fold_index] = values.reshape(-1)
                else:
                    self._bauer_scores[index, fold_index] = float(values.mean())
        best_values = best.detach().cpu().reshape(-1)
        self._bauer_orders[fold_index] = (
            int(best_values[0])
            if best_values.numel() == 1
            else tuple(int(value) for value in best_values)
        )

    @staticmethod
    def _expand(matrix, batch):
        """Broadcast a system matrix over observation trajectories."""
        if matrix.ndim == 2:
            matrix = matrix.unsqueeze(0)
        if matrix.shape[0] == 1 and batch > 1:
            matrix = matrix.expand(batch, *matrix.shape[1:])
        if matrix.shape[0] != batch:
            raise ValueError("system batch dimension does not match observations")
        return matrix

    def _innovation_errors(self, estimator, observations, fold):
        """Compute validation innovations under rolling or recursive updates."""
        system = estimator.system_
        batch = observations.shape[0]
        transition = self._expand(system.transition, batch)
        observation = self._expand(system.observation, batch)
        gain = self._expand(system.gain, batch)
        state = torch.zeros(
            batch,
            transition.shape[-1],
            dtype=observations.dtype,
            device=observations.device,
        )
        held_out = []
        for time_index in range(fold.test_stop):
            prediction = torch.einsum("bmd,bd->bm", observation, state)
            innovation = observations[:, time_index] - prediction
            if time_index >= fold.test_start:
                held_out.append(innovation)
            in_training = time_index < fold.train_stop
            in_gap = fold.train_stop <= time_index < fold.test_start
            use_observation = (
                in_training
                or (in_gap and self.gap_mode == "warmup")
                or (
                    time_index >= fold.test_start
                    and self.prediction_mode == "rolling"
                )
            )
            # Innovations form: z_{t+1} = A z_t + K epsilon_t. In recursive
            # validation or embargo gaps, the K epsilon_t term is omitted.
            state = torch.einsum("bij,bj->bi", transition, state)
            if use_observation:
                state = state + torch.einsum("bdm,bm->bd", gain, innovation)
        return torch.stack(held_out, dim=1)

    def _score(self, errors, covariance):
        """Return held-out loss without mixing independent trajectories."""
        independent_batch = self.mode == "independent" and errors.shape[0] > 1
        if self.scoring == "rmse":
            if independent_batch:
                values = torch.sqrt(errors.square().mean(dim=(1, 2)))
                return values.detach().cpu().numpy().astype(float, copy=False)
            return float(torch.sqrt(errors.square().mean()))
        covariance = self._expand(covariance, errors.shape[0])
        chol, _ = stable_cholesky(covariance, jitter=1e-10)
        solved = torch.cholesky_solve(
            errors.unsqueeze(-1), chol[:, None]
        ).squeeze(-1)
        logdet = 2.0 * torch.log(
            torch.diagonal(chol, dim1=-2, dim2=-1)
        ).sum(-1)
        # Gaussian negative log-likelihood: 1/2(e'V^-1e + log|V| + n log2pi).
        values = 0.5 * (
            (errors * solved).sum(-1)
            + logdet[:, None]
            + errors.shape[-1] * np.log(2.0 * np.pi)
        )
        if independent_batch:
            return values.mean(dim=1).detach().cpu().numpy().astype(float, copy=False)
        return float(values.mean())

    def _fit_and_score_fold(self, data, order, fold):
        """Compatibility hook fitting and scoring one order on one fold."""
        orders = tuple(sorted({int(value) for value in self.orders}))
        workspace = self._prepare_fold(data, fold, orders)
        estimator = workspace["decomposition"].fit_order(
            order, covariance=self.covariance
        )
        errors = self._innovation_errors(
            estimator, workspace["centered"], fold
        )
        return self._score(errors, estimator.innovation_covariance_)

    def _evaluate_candidate(self, workspace, order, fold):
        """Fit one truncated candidate and evaluate its held-out loss."""
        # Every dimension reuses the same fold-level CVA basis; only truncation
        # and the A,C,K,V regressions are candidate-specific.
        if (
            type(self)._fit_and_score_fold
            is not StateSpaceOrderSearchCV._fit_and_score_fold
        ):
            return self._fit_and_score_fold(
                workspace["centered"], order, fold
            )
        estimator = workspace["decomposition"].fit_order(
            order, covariance=self.covariance
        )
        errors = self._innovation_errors(
            estimator, workspace["centered"], fold
        )
        return self._score(errors, estimator.innovation_covariance_)

    def _finalize_diagnostics(self, orders):
        """Expose Bauer curves and fold means without losing batch axes."""
        del orders
        self.bauer_scores_ = self._bauer_scores
        finite = np.isfinite(self._bauer_scores)
        counts = finite.sum(axis=-1)
        totals = np.where(finite, self._bauer_scores, 0.0).sum(axis=-1)
        self.mean_bauer_scores_ = np.divide(
            totals,
            counts,
            out=np.full(counts.shape, np.nan, dtype=float),
            where=counts > 0,
        )
        self.bauer_order_per_fold_ = np.asarray(self._bauer_orders, dtype=object)

    def _fit_larimore(self, data, order, *, mode=None):
        """Fit one fixed-order Larimore model with this search configuration."""
        return LarimoreStateSpace(
            n_states=order,
            past_horizon=self.past_horizon,
            future_horizon=self.future_horizon,
            ridge=self.ridge,
            covariance=self.covariance,
            mode=self.mode if mode is None else mode,
            device=self.device,
            dtype=self.dtype,
        ).fit(data)

    def _refit_best(self, data, order):
        """Refit selected latent dimension(s) on all observations."""
        selected = np.asarray(order)
        if selected.ndim == 0:
            return self._fit_larimore(data, int(selected))
        if selected.shape != (data.shape[0],):
            raise ValueError("selected orders must match the batch dimension")
        return tuple(
            self._fit_larimore(data[index : index + 1], int(value), mode="independent")
            for index, value in enumerate(selected)
        )

    @staticmethod
    def _order_slice(values, index):
        """Return one candidate's scalar or per-batch values."""
        if values.ndim == 1:
            return float(values[index])
        return values[..., index].copy()

    @staticmethod
    def _fold_order_slice(values, index):
        """Return one candidate's fold values without losing batch axes."""
        if values.ndim == 2:
            return tuple(values[index])
        return values[..., index, :].copy()

    def fit(self, X, y=None):
        """Evaluate every candidate dimension on every temporal fold.

        Batched ``mode="independent"`` searches return one latent dimension
        per trajectory. If ``refit=True`` and selected dimensions differ,
        ``best_estimator_`` is a tuple with one Larimore estimator per
        trajectory; ``best_estimators_`` is provided as an explicit alias.
        """
        del y
        data = self._normalise_observations(X)
        summary = self._run_temporal_search(data)
        if self.refit and self.mode == "independent" and data.shape[0] > 1:
            self.best_estimators_ = self.best_estimator_
        scores = tuple(
            StateSpaceOrderScore(
                order=order,
                mean_score=self._order_slice(summary.mean_scores, index),
                standard_error=self._order_slice(summary.standard_errors, index),
                fold_scores=self._fold_order_slice(summary.fold_scores, index),
                failed_folds=self._order_slice(summary.failed_folds, index),
                mean_bauer_score=self._order_slice(self.mean_bauer_scores_, index),
                fold_bauer_scores=self._fold_order_slice(self.bauer_scores_, index),
            )
            for index, order in enumerate(summary.orders)
        )
        self.result_ = StateSpaceOrderSearchResult(
            summary.best_order,
            scores,
            self.scoring,
            self.selection_rule,
        )
        self.cv_results_ = self.result_.as_records()
        return self
