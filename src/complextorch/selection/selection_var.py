"""VAR lag-order selection by information criteria or temporal CV."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import torch
from sklearn.base import BaseEstimator

from ..linalg import spd_logdet, stable_cholesky
from ..var import VAR
from ._temporal import EpochTimeSeriesSplit, _TemporalOrderSearchCV
from .criteria import score_information_criteria


@dataclass(frozen=True)
class VARInformationCriteriaResult:
    """AIC, BIC, and HQC curves and their minimizing VAR orders."""

    p_aic: int | np.ndarray
    p_bic: int | np.ndarray
    p_hqc: int | np.ndarray
    aic: np.ndarray
    bic: np.ndarray
    hqc: np.ndarray
    loglik: np.ndarray
    orders: tuple[int, ...]


class VAROrderSelectionIC(BaseEstimator):
    r"""Select VAR lag order with in-sample AIC, BIC, and HQC.

    Each candidate lag :math:`p` is fitted independently, and the Gaussian
    residual likelihood is combined with the corresponding information-
    criterion penalty. ``refit`` selects which minimizing criterion determines
    ``best_order_`` and ``best_estimator_``; this selector is distinct from
    temporal cross-validation in :class:`VAROrderSearchCV`.
    """

    def __init__(
        self,
        orders: Iterable[int] = range(1, 11),
        *,
        solver: str = "lwr",
        mode: Literal["independent", "pooled"] = "pooled",
        hurvich_tsai: bool = False,
        device: str = "auto",
        dtype: str = "float64",
        refit: str | None = "hqc",
    ):
        """Initialize VAR information-criterion selection.

        Parameters
        ----------
        orders
            Candidate positive autoregressive orders.
        solver
            Numerical VAR solver forwarded unchanged to :class:`VAR`.
        mode
            ``"pooled"`` fits one model across trajectories while preserving
            trajectory boundaries. ``"independent"`` fits one model per batch
            element and returns one criterion curve and selected order per
            trajectory.
        hurvich_tsai
            Apply the Hurvich--Tsai small-sample correction to AIC.
        device
            Torch device or ``"auto"``.
        dtype
            Torch floating-point dtype name or object.
        refit
            Criterion used to choose and expose ``best_estimator_``. One of
            ``"aic"``, ``"bic"``, ``"hqc"``, or ``None`` to skip refitting.
        """
        self.orders = orders
        self.solver = solver
        self.mode = mode
        self.hurvich_tsai = hurvich_tsai
        self.device = device
        self.dtype = dtype
        self.refit = refit

    def fit(self, X: np.ndarray | torch.Tensor, y=None):
        """Fit every candidate order and compute AIC, BIC, and HQC curves."""
        del y
        orders = tuple(int(value) for value in self.orders)
        if not orders or min(orders) < 1:
            raise ValueError("orders must be positive")
        if self.mode not in {"independent", "pooled"}:
            raise ValueError("mode must be 'independent' or 'pooled'")
        data = torch.as_tensor(X)
        input_was_batched = data.ndim == 3
        if data.ndim == 2:
            data = data.unsqueeze(0)
        if data.ndim != 3:
            raise ValueError("X must have shape (time,n) or (batch,time,n)")
        n_trials, n_times, n_variables = data.shape
        independent_batch = self.mode == "independent" and n_trials > 1
        if independent_batch and self.refit is not None:
            raise ValueError(
                "refit must be None for batched mode='independent' because "
                "selected trajectories may have heterogeneous VAR orders"
            )
        loglik_by_order = []
        fitted = {}
        for order in orders:
            estimator = VAR(
                order=order,
                solver=self.solver,
                covariance="unbiased",
                mode=self.mode if input_was_batched else "independent",
                device=self.device,
                dtype=self.dtype,
                stability="ignore",
            ).fit(data)
            fitted[order] = estimator
            covariance = estimator.noise_covariance_
            logdet = spd_logdet(covariance)
            per_model = -0.5 * (
                n_variables * np.log(2.0 * np.pi) + logdet + n_variables
            )
            if independent_batch:
                loglik_by_order.append(
                    per_model.detach().cpu().numpy().astype(float, copy=False)
                )
            else:
                loglik_by_order.append(float(per_model.reshape(-1)[0]))
        if independent_batch:
            likelihood = np.stack(loglik_by_order, axis=1)
            parameter_counts = np.asarray(
                [order * n_variables * n_variables for order in orders], dtype=float
            )[None, :]
            observation_counts = np.asarray(
                [n_times - order for order in orders], dtype=float
            )[None, :]
        else:
            likelihood = np.asarray(loglik_by_order)
            parameter_counts = np.asarray(
                [order * n_variables * n_variables for order in orders], dtype=float
            )
            multiplier = n_trials if self.mode == "pooled" else 1
            observation_counts = np.asarray(
                [multiplier * (n_times - order) for order in orders], dtype=float
            )
        criteria = score_information_criteria(
            likelihood,
            parameter_counts,
            observation_counts,
            likelihood="mean",
            scale="per_observation",
            hurvich_tsai=self.hurvich_tsai,
        )
        aic, bic, hqc = criteria.aic, criteria.bic, criteria.hqc
        if independent_batch:
            p_aic = np.asarray(orders)[np.nanargmin(aic, axis=1)]
            p_bic = np.asarray(orders)[np.nanargmin(bic, axis=1)]
            p_hqc = np.asarray(orders)[np.nanargmin(hqc, axis=1)]
        else:
            p_aic = orders[int(np.nanargmin(aic))]
            p_bic = orders[int(np.nanargmin(bic))]
            p_hqc = orders[int(np.nanargmin(hqc))]
        result = VARInformationCriteriaResult(
            p_aic, p_bic, p_hqc, aic, bic, hqc, likelihood, orders
        )
        self.result_ = result
        self.p_aic_ = result.p_aic
        self.p_bic_ = result.p_bic
        self.p_hqc_ = result.p_hqc
        self.aic_ = aic
        self.bic_ = bic
        self.hqc_ = hqc
        self.loglik_ = likelihood
        self.candidate_estimators_ = fitted
        if self.refit is not None:
            key = self.refit.lower()
            if key not in {"aic", "bic", "hqc"}:
                raise ValueError("refit must be None, 'aic', 'bic' or 'hqc'")
            self.best_order_ = getattr(result, f"p_{key}")
            self.best_estimator_ = fitted[self.best_order_]
        return self


@dataclass(frozen=True)
class VAROrderScore:
    """Held-out and information-criterion diagnostics for one VAR order.

    Scalar fields remain scalars for a single series or pooled search. For
    batched ``mode="independent"`` they are arrays with one value per
    trajectory; fold-level fields have shape ``(batch, n_folds)``.
    """

    order: int
    mean_score: float | np.ndarray
    standard_error: float | np.ndarray
    fold_scores: tuple[float, ...] | np.ndarray
    failed_folds: int | np.ndarray
    mean_train_aic: float | np.ndarray
    mean_train_bic: float | np.ndarray
    mean_train_hqc: float | np.ndarray
    fold_train_aic: tuple[float, ...] | np.ndarray
    fold_train_bic: tuple[float, ...] | np.ndarray
    fold_train_hqc: tuple[float, ...] | np.ndarray


@dataclass(frozen=True)
class VAROrderSearchResult:
    """Immutable summary returned by temporal VAR order search."""

    best_order: int | np.ndarray
    scores: tuple[VAROrderScore, ...]
    scoring: str
    selection_rule: str

    def as_records(self):
        """Return records suitable for tabular serialization."""
        return [score.__dict__.copy() for score in self.scores]


class VAROrderSearchCV(_TemporalOrderSearchCV):
    """Select VAR lag order by leakage-safe temporal cross-validation.

    For batched ``mode="independent"`` input, every trajectory retains its
    own validation-loss curve and selected order. ``mode="pooled"`` retains
    the historical behavior of selecting one common order across trajectories.

    ``prediction_mode='rolling'`` updates predictions with each observed
    validation sample after it is scored. ``'recursive'`` never consumes held-
    out observations. ``gap_mode='warmup'`` allows unscored gap observations to
    initialize prediction; ``'embargo'`` excludes them and propagates only
    model predictions through the gap.
    """

    def __init__(
        self,
        orders: Iterable[int],
        *,
        cv: EpochTimeSeriesSplit | None = None,
        scoring: Literal["nll", "rmse"] = "nll",
        selection_rule: Literal["best", "one_se"] = "one_se",
        alpha: float = 0.0,
        fit_intercept: bool = True,
        mode: Literal["independent", "pooled"] = "independent",
        solver: str = "auto",
        device: str = "auto",
        dtype: str = "float64",
        prediction_mode: Literal["rolling", "recursive"] = "rolling",
        gap_mode: Literal["warmup", "embargo"] = "warmup",
        refit: bool = True,
        hurvich_tsai: bool = False,
    ):
        """Initialize temporal VAR order-search settings."""
        self._set_temporal_search_parameters(
            orders=orders,
            cv=cv,
            scoring=scoring,
            selection_rule=selection_rule,
            prediction_mode=prediction_mode,
            gap_mode=gap_mode,
            refit=refit,
        )
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.mode = mode
        self.solver = solver
        self.device = device
        self.dtype = dtype
        self.hurvich_tsai = hurvich_tsai

    def _score_batch_shape(self, data):
        """Preserve one validation curve per independent batch trajectory."""
        if self.mode == "independent" and data.shape[0] > 1:
            return (data.shape[0],)
        return ()

    def _begin_diagnostics(self, n_orders, n_folds):
        """Allocate training-fold AIC, BIC, and HQC arrays."""
        shape = self._score_batch_shape_ + (n_orders, n_folds)
        self._aic = np.full(shape, np.nan)
        self._bic = np.full(shape, np.nan)
        self._hqc = np.full(shape, np.nan)

    def _prepare_fold(self, data, fold, orders):
        """Return the raw series and training prefix for one fold."""
        return {
            "data": data,
            "training": data[:, : fold.train_stop],
            "fold": fold,
            "orders": orders,
        }

    def _fit_var(self, training, order):
        """Fit one fixed-order VAR exclusively on the training prefix."""
        return VAR(
            order=order,
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            mode=self.mode,
            solver=self.solver,
            covariance="unbiased",
            device=self.device,
            dtype=self.dtype,
            stability="ignore",
        ).fit(training)

    @staticmethod
    def _expand(matrix, batch):
        """Broadcast a fitted parameter tensor over observation trajectories."""
        if matrix.shape[0] == 1 and batch > 1:
            return matrix.expand(batch, *matrix.shape[1:])
        return matrix

    def _predict_block(self, estimator, data, fold):
        """Predict a validation block under configured update semantics."""
        batch = data.shape[0]
        coefficients = self._expand(estimator.coef_, batch)
        intercept = self._expand(estimator.intercept_, batch)
        order = estimator.order
        history = data[:, : fold.train_stop].to(coefficients)
        state = history[:, -order:].clone()
        predictions = []
        for time_index in range(fold.train_stop, fold.test_stop):
            prediction = intercept.clone()
            for lag in range(order):
                prediction = prediction + torch.einsum(
                    "bij,bj->bi",
                    coefficients[:, lag],
                    state[:, -(lag + 1)],
                )
            in_gap = time_index < fold.test_start
            if not in_gap:
                predictions.append(prediction)
            use_observation = (
                (in_gap and self.gap_mode == "warmup")
                or ((not in_gap) and self.prediction_mode == "rolling")
            )
            next_value = (
                data[:, time_index].to(prediction)
                if use_observation
                else prediction
            )
            state = torch.cat((state[:, 1:], next_value[:, None]), dim=1)
        return torch.stack(predictions, dim=1)

    def _score(self, errors, covariance):
        """Return held-out RMSE or Gaussian NLL without mixing trajectories."""
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
        values = 0.5 * (
            (errors * solved).sum(-1)
            + 2.0
            * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)[
                :, None
            ]
            + errors.shape[-1] * np.log(2.0 * np.pi)
        )
        if independent_batch:
            return (
                values.mean(dim=1).detach().cpu().numpy().astype(float, copy=False)
            )
        return float(values.mean())

    def _fold_diagnostics(self, workspace, orders, fold_index):
        """Store fold and candidate indices needed by IC diagnostics."""
        workspace["fold_index"] = fold_index
        workspace["orders"] = orders

    def _evaluate_candidate(self, workspace, order, fold):
        """Fit and score one candidate and record its training IC values."""
        estimator = self._fit_var(workspace["training"], order)
        prediction = self._predict_block(estimator, workspace["data"], fold)
        target = workspace["data"][
            :, fold.test_start : fold.test_stop
        ].to(prediction)
        order_index = workspace["orders"].index(order)
        self._store_ic(
            estimator,
            workspace["training"],
            order,
            order_index,
            workspace["fold_index"],
        )
        return self._score(target - prediction, estimator.noise_covariance_)

    def _store_ic(self, estimator, training, order, order_index, fold_index):
        """Compute and store training-only AIC, BIC, and HQC diagnostics."""
        n_trials, n_times, n_variables = training.shape
        logdet = spd_logdet(estimator.noise_covariance_)
        per_model = -0.5 * (
            n_variables * np.log(2.0 * np.pi) + logdet + n_variables
        )
        independent_batch = self.mode == "independent" and n_trials > 1
        if independent_batch:
            loglik = per_model.detach().cpu().numpy().astype(float, copy=False)
            n_parameters = order * n_variables * n_variables
            n_observations = n_times - order
        else:
            loglik = float(per_model.mean())
            n_parameters = order * n_variables * n_variables
            n_observations = (
                n_trials * (n_times - order)
                if self.mode == "pooled"
                else n_times - order
            )
        criteria = score_information_criteria(
            loglik,
            n_parameters,
            n_observations,
            likelihood="mean",
            scale="per_observation",
            hurvich_tsai=self.hurvich_tsai,
        )
        aic, bic, hqc = criteria.aic, criteria.bic, criteria.hqc
        self._aic[..., order_index, fold_index] = aic
        self._bic[..., order_index, fold_index] = bic
        self._hqc[..., order_index, fold_index] = hqc

    def _finalize_diagnostics(self, orders):
        """Expose fold-level and mean IC diagnostics as fitted attributes."""
        del orders
        self.train_aic_ = self._aic
        self.train_bic_ = self._bic
        self.train_hqc_ = self._hqc
        self.mean_train_aic_ = np.nanmean(self._aic, axis=-1)
        self.mean_train_bic_ = np.nanmean(self._bic, axis=-1)
        self.mean_train_hqc_ = np.nanmean(self._hqc, axis=-1)

    def _refit_best(self, data, order):
        """Refit selected VAR order(s) on all observations.

        A scalar order returns one estimator. Batched independent searches may
        select heterogeneous orders, in which case one fixed-order estimator
        is refitted per trajectory and returned as a tuple.
        """
        selected = np.asarray(order)
        if selected.ndim == 0:
            return self._fit_var(data, int(selected))
        if selected.shape != (data.shape[0],):
            raise ValueError("selected orders must match the batch dimension")
        return tuple(
            self._fit_var(data[index : index + 1], int(value))
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
        """Evaluate every candidate VAR order on every temporal fold.

        Batched ``mode="independent"`` searches return one order per
        trajectory. If ``refit=True`` and selected orders differ,
        ``best_estimator_`` is a tuple containing one fixed-order VAR per
        trajectory; ``best_estimators_`` is provided as an explicit alias.
        """
        del y
        data = self._normalise_observations(X)
        summary = self._run_temporal_search(data)
        if self.refit and self.mode == "independent" and data.shape[0] > 1:
            self.best_estimators_ = self.best_estimator_
        scores = tuple(
            VAROrderScore(
                order=order,
                mean_score=self._order_slice(summary.mean_scores, index),
                standard_error=self._order_slice(summary.standard_errors, index),
                fold_scores=self._fold_order_slice(summary.fold_scores, index),
                failed_folds=self._order_slice(summary.failed_folds, index),
                mean_train_aic=self._order_slice(self.mean_train_aic_, index),
                mean_train_bic=self._order_slice(self.mean_train_bic_, index),
                mean_train_hqc=self._order_slice(self.mean_train_hqc_, index),
                fold_train_aic=self._fold_order_slice(self.train_aic_, index),
                fold_train_bic=self._fold_order_slice(self.train_bic_, index),
                fold_train_hqc=self._fold_order_slice(self.train_hqc_, index),
            )
            for index, order in enumerate(summary.orders)
        )
        self.result_ = VAROrderSearchResult(
            summary.best_order,
            scores,
            self.scoring,
            self.selection_rule,
        )
        self.cv_results_ = self.result_.as_records()
        return self
