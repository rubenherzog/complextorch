r"""Temporal cross-validation for latent state-space dimension.

The candidate latent dimension :math:`r` is treated as a global
hyperparameter. Every candidate is fitted on every temporal training fold and
scored on the corresponding held-out block. The selected dimension minimizes
aggregated held-out prediction loss; fold-wise Bauer SVC curves are exposed as
training-only diagnostics and never determine ``best_order_``.

For an innovations-form model

.. math::

   z_{t+1}=Az_t+K\varepsilon_t,\qquad
   y_t=Cz_t+\varepsilon_t,\qquad
   \varepsilon_t\sim\mathcal N(0,V),

the default score is the mean held-out Gaussian negative log likelihood

.. math::

   \mathcal L(r)=\frac{1}{N}\sum_t\frac{1}{2}
   \left[n_y\log(2\pi)+\log\det V+
   \varepsilon_t^\top V^{-1}\varepsilon_t\right].

References
----------
- Larimore, W. E. (1990, 1996), canonical variate analysis system
  identification.
- Bauer, D. (2001). Order estimation for subspace methods. *Automatica*,
  37(10), 1561--1573.
- Racine, J. (2000). Consistent cross-validatory model-selection for dependent
  data. *Journal of Econometrics*, 99(1), 39--61.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Literal

import numpy as np
import torch
from sklearn.base import BaseEstimator

from ._typing import ArrayLike
from .linalg import stable_cholesky
from .selection import EpochTimeSeriesSplit, StateSpaceOrderSelection, TemporalFold
from .state_space import LarimoreStateSpace


@dataclass(frozen=True)
class StateSpaceOrderScore:
    """Held-out and training-diagnostic results for one latent dimension.

    Parameters
    ----------
    order
        Candidate latent state dimension.
    mean_score
        Validation-size-weighted mean held-out loss across successful folds.
    standard_error
        Standard error of the successful fold-level losses.
    fold_scores
        One held-out loss per temporal fold; failed folds are ``inf``.
    failed_folds
        Number of folds for which fitting or scoring failed.
    mean_bauer_score
        Mean finite training-fold Bauer SVC value for this order.
    fold_bauer_scores
        Training-fold Bauer SVC values; unavailable values are ``nan``.
    """

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

    def as_records(self) -> list[dict[str, object]]:
        """Return records suitable for a dataframe or serialization."""

        return [
            {
                "order": score.order,
                "mean_score": score.mean_score,
                "standard_error": score.standard_error,
                "fold_scores": score.fold_scores,
                "failed_folds": score.failed_folds,
                "mean_bauer_score": score.mean_bauer_score,
                "fold_bauer_scores": score.fold_bauer_scores,
            }
            for score in self.scores
        ]


class StateSpaceOrderSearchCV(BaseEstimator):
    r"""Select one global latent dimension by temporal cross-validation.

    Every order in ``orders`` is fitted independently on every training fold
    with :class:`~complextorch.state_space.LarimoreStateSpace`. The same order
    is therefore compared across all folds. The final order is selected from
    aggregated held-out loss rather than by voting over fold-wise selections.

    A :class:`~complextorch.selection.StateSpaceOrderSelection` Bauer curve is
    also computed on each training fold. These curves and the corresponding
    fold-wise Bauer choices are diagnostics only.

    Parameters
    ----------
    orders
        Candidate positive latent state dimensions.
    past_horizon
        Number of block rows in the past Hankel matrix.
    future_horizon
        Number of block rows in the future Hankel matrix. Defaults to
        ``past_horizon``.
    cv
        Expanding-window temporal splitter. The splitter gap should be chosen
        to match the desired temporal embargo. Hankel matrices are always
        reconstructed inside each training fold, so they never cross the
        train/validation boundary.
    scoring
        ``"nll"`` for Gaussian innovation negative log likelihood or
        ``"rmse"`` for one-step innovation root mean-square error.
    selection_rule
        ``"best"`` selects the minimum mean loss. ``"one_se"`` selects the
        smallest order within one standard error of that minimum.
    ridge
        Non-negative regularization used by Larimore Cholesky whitening.
    covariance
        Innovation covariance normalization passed to Larimore fitting.
    mode
        ``"pooled"`` estimates one common model across independent
        trajectories. ``"independent"`` fits one model per trajectory. In
        both cases a single global order is selected by aggregating held-out
        losses across trajectories and folds.
    device, dtype
        Torch execution device and floating-point dtype.
    refit
        Whether to fit ``best_estimator_`` on all observations.
    bauer_diagnostics
        Whether to compute fold-wise training-only Bauer SVC curves.

    Attributes
    ----------
    best_order_
        Selected global latent state dimension.
    best_estimator_
        Larimore estimator refitted on all data when ``refit=True``.
    cv_results_
        Record representation of all candidate results.
    fold_scores_
        Array with shape ``(n_orders, n_folds)``.
    mean_test_scores_, standard_error_
        Aggregated held-out loss and fold-level standard errors.
    bauer_scores_
        Training-fold Bauer SVC values with shape ``(n_orders, n_folds)``.
    bauer_order_per_fold_
        Bauer-selected order for each training fold. For independent mode this
        is an object array because each trajectory may select a distinct order.
    failure_messages_
        Mapping ``(order, fold_index)`` to the fitting/scoring exception text.

    Notes
    -----
    Each trajectory is centered using its training-fold mean. During scoring,
    the innovations state is warmed chronologically with observations before
    the validation block. Validation observations are used only after they
    become past observations, which gives rolling one-step-ahead predictions.
    No state transition or Hankel column is ever constructed across trajectory
    boundaries.
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
        refit: bool = True,
        bauer_diagnostics: bool = True,
    ):
        self.orders = tuple(int(order) for order in orders)
        self.past_horizon = past_horizon
        self.future_horizon = future_horizon
        self.cv = cv or EpochTimeSeriesSplit()
        self.scoring = scoring
        self.selection_rule = selection_rule
        self.ridge = ridge
        self.covariance = covariance
        self.mode = mode
        self.device = device
        self.dtype = dtype
        self.refit = refit
        self.bauer_diagnostics = bauer_diagnostics

    @staticmethod
    def _normalise(X: ArrayLike) -> torch.Tensor:
        """Normalize observations to ``(batch, time, variables)``."""

        data = torch.as_tensor(X)
        if data.ndim == 2:
            data = data.unsqueeze(0)
        if data.ndim != 3:
            raise ValueError(
                "X must have shape (time, variables) or "
                "(batch, time, variables)"
            )
        if not torch.isfinite(data).all():
            raise ValueError("X must contain only finite values")
        return data

    def _validate_settings(self) -> tuple[int, ...]:
        """Validate constructor settings and return sorted unique orders."""

        orders = tuple(sorted(set(self.orders)))
        if not orders or orders[0] < 1:
            raise ValueError("orders must contain positive integers")
        future_horizon = (
            self.past_horizon
            if self.future_horizon is None
            else self.future_horizon
        )
        if self.past_horizon < 1 or future_horizon < 1:
            raise ValueError("past_horizon and future_horizon must be positive")
        if self.ridge < 0:
            raise ValueError("ridge must be non-negative")
        if self.scoring not in {"nll", "rmse"}:
            raise ValueError("scoring must be 'nll' or 'rmse'")
        if self.selection_rule not in {"best", "one_se"}:
            raise ValueError("selection_rule must be 'best' or 'one_se'")
        if self.covariance not in {"mle", "unbiased"}:
            raise ValueError("covariance must be 'mle' or 'unbiased'")
        if self.mode not in {"pooled", "independent"}:
            raise ValueError("mode must be 'pooled' or 'independent'")
        return orders

    @staticmethod
    def _expand_system_matrix(matrix: torch.Tensor, batch: int) -> torch.Tensor:
        """Add or expand a leading system batch dimension."""

        if matrix.ndim == 2:
            matrix = matrix.unsqueeze(0)
        if matrix.shape[0] == 1 and batch > 1:
            matrix = matrix.expand(batch, *matrix.shape[1:])
        if matrix.shape[0] != batch:
            raise ValueError("system batch dimension does not match observations")
        return matrix

    def _innovation_errors(
        self,
        estimator: LarimoreStateSpace,
        observations: torch.Tensor,
        fold: TemporalFold,
    ) -> torch.Tensor:
        """Return rolling one-step innovations on one validation block."""

        system = estimator.system_
        batch = observations.shape[0]
        transition = self._expand_system_matrix(system.transition, batch)
        observation = self._expand_system_matrix(system.observation, batch)
        gain = self._expand_system_matrix(system.gain, batch)
        state = torch.zeros(
            (batch, transition.shape[-1]),
            dtype=observations.dtype,
            device=observations.device,
        )
        held_out = []
        for time_index in range(fold.test_stop):
            prediction = torch.einsum("bmd,bd->bm", observation, state)
            innovation = observations[:, time_index] - prediction
            if time_index >= fold.test_start:
                held_out.append(innovation)
            # The current observation enters the state only after its one-step
            # prediction error has been formed, preventing look-ahead leakage.
            state = torch.einsum("bij,bj->bi", transition, state) + torch.einsum(
                "bdm,bm->bd", gain, innovation
            )
        return torch.stack(held_out, dim=1)

    def _score_errors(
        self,
        errors: torch.Tensor,
        innovation_covariance: torch.Tensor,
    ) -> float:
        """Compute the configured held-out score."""

        if self.scoring == "rmse":
            return float(torch.sqrt(torch.mean(errors.square())))
        covariance = self._expand_system_matrix(
            innovation_covariance, errors.shape[0]
        )
        cholesky, _ = stable_cholesky(covariance, jitter=1e-10)
        solved = torch.cholesky_solve(
            errors.unsqueeze(-1), cholesky[:, None]
        ).squeeze(-1)
        logdet = 2.0 * torch.log(
            torch.diagonal(cholesky, dim1=-2, dim2=-1)
        ).sum(-1)
        n_variables = errors.shape[-1]
        nll = 0.5 * (
            (errors * solved).sum(-1)
            + logdet[:, None]
            + n_variables * np.log(2.0 * np.pi)
        )
        return float(nll.mean())

    def _fit_and_score_fold(
        self,
        data: torch.Tensor,
        order: int,
        fold: TemporalFold,
    ) -> float:
        """Fit one fixed-order Larimore model and score its validation block."""

        training_mean = data[:, : fold.train_stop].mean(dim=1, keepdim=True)
        centered = data - training_mean
        estimator = LarimoreStateSpace(
            n_states=order,
            past_horizon=self.past_horizon,
            future_horizon=self.future_horizon,
            ridge=self.ridge,
            covariance=self.covariance,
            mode=self.mode,
            device=self.device,
            dtype=self.dtype,
        ).fit(centered[:, : fold.train_stop])
        scoring_data = centered[:, : fold.test_stop].to(
            device=estimator.transition_.device,
            dtype=estimator.transition_.dtype,
        )
        errors = self._innovation_errors(estimator, scoring_data, fold)
        return self._score_errors(errors, estimator.innovation_covariance_)

    def _bauer_fold(
        self,
        training: torch.Tensor,
        orders: tuple[int, ...],
    ) -> tuple[np.ndarray, object]:
        """Compute one training-only Bauer diagnostic curve."""

        selector = StateSpaceOrderSelection(
            past_horizon=self.past_horizon,
            future_horizon=self.future_horizon,
            min_order=min(orders),
            mode=self.mode,
            ridge=self.ridge,
            device=self.device,
            dtype=self.dtype,
            refit=False,
        ).fit(training)
        selector_orders = torch.as_tensor(selector.orders_).detach().cpu()
        criterion = torch.as_tensor(selector.criterion_).detach().cpu()
        curve = np.full(len(orders), np.nan, dtype=float)
        for index, order in enumerate(orders):
            matches = torch.nonzero(selector_orders == order, as_tuple=False)
            if matches.numel() == 0:
                continue
            criterion_index = int(matches[0, -1])
            value = criterion[..., criterion_index]
            curve[index] = float(value.mean())
        best = torch.as_tensor(selector.best_order_).detach().cpu()
        if best.ndim == 0:
            return curve, int(best)
        return curve, tuple(int(value) for value in best.reshape(-1))

    @staticmethod
    def _finite_mean(values: np.ndarray) -> float:
        """Return the mean of finite values or ``nan`` when none exist."""

        finite = values[np.isfinite(values)]
        return float(finite.mean()) if finite.size else float("nan")

    def fit(self, X: ArrayLike, y=None):
        """Evaluate every candidate dimension on every temporal fold.

        Parameters
        ----------
        X
            Observations with shape ``(time, variables)`` or
            ``(batch, time, variables)``.
        y
            Unused scikit-learn compatibility target.

        Returns
        -------
        StateSpaceOrderSearchCV
            Fitted search estimator.
        """

        del y
        orders = self._validate_settings()
        data = self._normalise(X)
        future_horizon = (
            self.past_horizon
            if self.future_horizon is None
            else self.future_horizon
        )
        # The first training fold must support both block-Hankel horizons and
        # the largest requested state dimension.
        minimum_training = self.past_horizon + future_horizon + max(orders) + 2
        folds = tuple(self.cv.split(data.shape[1], min_order=minimum_training))
        if not folds:
            raise ValueError("cv produced no folds")

        fold_scores = np.full((len(orders), len(folds)), np.inf, dtype=float)
        bauer_scores = np.full((len(orders), len(folds)), np.nan, dtype=float)
        bauer_orders: list[object] = []
        failure_messages: dict[tuple[int, int], str] = {}

        for fold_index, fold in enumerate(folds):
            training_mean = data[:, : fold.train_stop].mean(dim=1, keepdim=True)
            centered_training = data[:, : fold.train_stop] - training_mean
            if self.bauer_diagnostics:
                try:
                    curve, selected = self._bauer_fold(centered_training, orders)
                    bauer_scores[:, fold_index] = curve
                    bauer_orders.append(selected)
                except (RuntimeError, ValueError, torch.linalg.LinAlgError) as error:
                    bauer_orders.append(None)
                    failure_messages[(-1, fold_index)] = f"Bauer diagnostic: {error}"
            else:
                bauer_orders.append(None)

            for order_index, order in enumerate(orders):
                try:
                    fold_scores[order_index, fold_index] = self._fit_and_score_fold(
                        data, order, fold
                    )
                except (RuntimeError, ValueError, torch.linalg.LinAlgError) as error:
                    failure_messages[(order, fold_index)] = str(error)

        results = []
        for order_index, order in enumerate(orders):
            scores = fold_scores[order_index]
            finite = scores[np.isfinite(scores)]
            mean_score = float(finite.mean()) if finite.size else float("inf")
            standard_error = (
                float(finite.std(ddof=1) / sqrt(finite.size))
                if finite.size > 1
                else 0.0
            )
            results.append(
                StateSpaceOrderScore(
                    order=order,
                    mean_score=mean_score,
                    standard_error=standard_error,
                    fold_scores=tuple(float(value) for value in scores),
                    failed_folds=int((~np.isfinite(scores)).sum()),
                    mean_bauer_score=self._finite_mean(bauer_scores[order_index]),
                    fold_bauer_scores=tuple(
                        float(value) for value in bauer_scores[order_index]
                    ),
                )
            )

        finite_results = [
            result for result in results if np.isfinite(result.mean_score)
        ]
        if not finite_results:
            raise RuntimeError("all candidate state dimensions failed")
        optimum = min(finite_results, key=lambda result: result.mean_score)
        if self.selection_rule == "best":
            best_order = optimum.order
        else:
            best_order = min(
                result.order
                for result in finite_results
                if result.mean_score
                <= optimum.mean_score + optimum.standard_error
            )

        self.result_ = StateSpaceOrderSearchResult(
            best_order=best_order,
            scores=tuple(results),
            scoring=self.scoring,
            selection_rule=self.selection_rule,
        )
        self.best_order_ = best_order
        self.orders_ = np.asarray(orders, dtype=int)
        self.folds_ = folds
        self.cv_results_ = self.result_.as_records()
        self.fold_scores_ = fold_scores
        self.mean_test_scores_ = np.asarray(
            [result.mean_score for result in results], dtype=float
        )
        self.standard_error_ = np.asarray(
            [result.standard_error for result in results], dtype=float
        )
        self.bauer_scores_ = bauer_scores
        self.mean_bauer_scores_ = np.asarray(
            [result.mean_bauer_score for result in results], dtype=float
        )
        self.bauer_order_per_fold_ = np.asarray(bauer_orders, dtype=object)
        self.failure_messages_ = failure_messages

        if self.refit:
            self.best_estimator_ = LarimoreStateSpace(
                n_states=best_order,
                past_horizon=self.past_horizon,
                future_horizon=self.future_horizon,
                ridge=self.ridge,
                covariance=self.covariance,
                mode=self.mode,
                device=self.device,
                dtype=self.dtype,
            ).fit(data)
        return self
