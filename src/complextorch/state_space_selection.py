r"""Temporal cross-validation for latent state-space dimension.

Every temporal training fold computes one Larimore CVA decomposition at the
maximum identifiable rank. Candidate dimensions truncate that common canonical
basis, estimate :math:`A,C,K,V` on training data, and are evaluated only on the
held-out block. Bauer SVC is reported from the same fold decomposition as a
training-only diagnostic.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Iterable, Literal

import numpy as np
import torch

from ._state_space_order import _bauer_svc, _block_hankel
from ._temporal_order_search import _TemporalOrderSearchCV
from .control import InnovationsStateSpace
from .linalg import stable_cholesky
from .selection import EpochTimeSeriesSplit
from .state_space import (
    LarimoreStateSpace,
    _fit_innovations_state_space_from_states,
    _larimore_decomposition,
    _normalise_ss_observations,
)


@dataclass(frozen=True)
class StateSpaceOrderScore:
    """Held-out and Bauer diagnostic results for one latent dimension."""

    order: int
    mean_score: float
    standard_error: float
    fold_scores: tuple[float, ...]
    failed_folds: int
    mean_bauer_score: float
    fold_bauer_scores: tuple[float, ...]


@dataclass(frozen=True)
class StateSpaceOrderSearchResult:
    """Immutable summary of temporal state-space order search."""

    best_order: int
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
        # Apply the covariance or whitening solve through its triangular Cholesky factor.
        self.whitened_past = torch.linalg.solve_triangular(
            self.cholesky_past.transpose(-1, -2),
            self.past_fit,
            upper=True,
        )

    def fit_order(self, order, *, covariance):
        """Truncate the shared canonical basis and estimate ``A,C,K,V``."""

        if order > self.correlations.shape[-1]:
            raise ValueError("state dimension exceeds identifiable subspace rank")
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
    r"""Select one global latent dimension by temporal cross-validation.

    ``prediction_mode='rolling'`` updates the innovations state with each
    validation observation after scoring it. ``'recursive'`` propagates only
    the fitted model during validation. ``gap_mode='warmup'`` consumes unscored
    gap observations, while ``'embargo'`` excludes them and propagates only
    :math:`z_{t+1}=Az_t` across the gap.

    A single Larimore CVA decomposition is computed per fold and shared across
    all candidate dimensions. Bauer SVC remains a training-only diagnostic.
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
        covariance: Literal["mle", "unbiased"] = "unbiased",
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

        self._bauer_scores = np.full((n_orders, n_folds), np.nan)
        self._bauer_orders = [None] * n_folds

    def _prepare_fold(self, data, fold, orders):
        """Center one fold and compute its reusable Larimore workspace."""

        values, _ = _normalise_ss_observations(
            data, device=self.device, dtype=self.dtype
        )
        training_mean = values[:, :fold.train_stop].mean(dim=1, keepdim=True)
        centered = values - training_mean
        future = (
            self.past_horizon
            if self.future_horizon is None
            else self.future_horizon
        )
        workspace = _LarimoreFoldWorkspace(
            centered[:, :fold.train_stop],
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
        lower = max(n_variables, min(orders))
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
        lower = max(n_variables, min(orders))
        if lower > decomposition.correlations.shape[-1]:
            return
        best, criterion = _bauer_svc(
            decomposition.correlations,
            n_variables,
            decomposition.n_columns
            * (decomposition.batch if self.mode == "pooled" else 1),
            min_order=lower,
        )
        candidates = torch.arange(lower, decomposition.correlations.shape[-1] + 1)
        for index, order in enumerate(orders):
            match = torch.nonzero(candidates == order, as_tuple=False)
            if match.numel():
                self._bauer_scores[index, fold_index] = float(
                    criterion[..., int(match[0])].mean()
                )
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
            state = torch.einsum("bij,bj->bi", transition, state)
            if use_observation:
                state = state + torch.einsum("bdm,bm->bd", gain, innovation)
        return torch.stack(held_out, dim=1)

    def _score(self, errors, covariance):
        """Return held-out innovation RMSE or Gaussian NLL."""

        if self.scoring == "rmse":
            return float(torch.sqrt(errors.square().mean()))
        covariance = self._expand(covariance, errors.shape[0])
        chol, _ = stable_cholesky(covariance, jitter=1e-10)
        solved = torch.cholesky_solve(
            errors.unsqueeze(-1), chol[:, None]
        ).squeeze(-1)
        logdet = 2.0 * torch.log(
            torch.diagonal(chol, dim1=-2, dim2=-1)
        ).sum(-1)
        values = 0.5 * (
            (errors * solved).sum(-1)
            + logdet[:, None]
            + errors.shape[-1] * np.log(2.0 * np.pi)
        )
        return float(values.mean())

    def _fit_and_score_fold(self, data, order, fold):
        """Compatibility hook fitting and scoring one order on one fold."""

        workspace = self._prepare_fold(
            data, fold, tuple(sorted(set(self.orders)))
        )
        estimator = workspace["decomposition"].fit_order(
            order, covariance=self.covariance
        )
        errors = self._innovation_errors(
            estimator, workspace["centered"], fold
        )
        return self._score(errors, estimator.innovation_covariance_)

    def _evaluate_candidate(self, workspace, order, fold):
        """Fit one truncated candidate and evaluate its held-out loss."""
        # Fit each candidate on every training window and aggregate held-out predictive loss across folds.

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
        """Expose Bauer curves and finite fold means as fitted attributes."""

        del orders
        self.bauer_scores_ = self._bauer_scores
        means = []
        for row in self._bauer_scores:
            finite = row[np.isfinite(row)]
            means.append(float(finite.mean()) if finite.size else float("nan"))
        self.mean_bauer_scores_ = np.asarray(means)
        self.bauer_order_per_fold_ = np.asarray(
            self._bauer_orders, dtype=object
        )

    def _refit_best(self, data, order):
        """Refit a Larimore model at the selected dimension on all data."""

        return LarimoreStateSpace(
            n_states=order,
            past_horizon=self.past_horizon,
            future_horizon=self.future_horizon,
            ridge=self.ridge,
            covariance=self.covariance,
            mode=self.mode,
            device=self.device,
            dtype=self.dtype,
        ).fit(data)

    def fit(self, X, y=None):
        """Evaluate every candidate dimension on every temporal fold."""

        del y
        summary = self._run_temporal_search(X)
        scores = tuple(
            StateSpaceOrderScore(
                order=order,
                mean_score=float(summary.mean_scores[index]),
                standard_error=float(summary.standard_errors[index]),
                fold_scores=tuple(summary.fold_scores[index]),
                failed_folds=int(summary.failed_folds[index]),
                mean_bauer_score=float(self.mean_bauer_scores_[index]),
                fold_bauer_scores=tuple(self.bauer_scores_[index]),
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
