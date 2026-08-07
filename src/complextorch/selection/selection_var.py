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


def _information_criteria(
    loglik: float | np.ndarray,
    n_parameters: int | np.ndarray,
    n_observations: int | np.ndarray,
    *,
    hurvich_tsai: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Return MVGC2-compatible per-observation AIC, BIC, and HQC.

    ``loglik`` is the Gaussian log-likelihood contribution per effective
    observation. With parameter-to-sample ratio :math:`k/N`, the criteria use
    the MVGC convention

    .. math::

       \mathrm{AIC}=-2\ell+2k/N,
       \qquad
       \mathrm{BIC}=-2\ell+(k/N)\log N,

    with HQC using :math:`2(k/N)\log\log N`. The optional Hurvich--Tsai
    correction multiplies the AIC penalty by :math:`N/(N-k-1)`.
    """
    likelihood = np.asarray(loglik, dtype=float)
    parameters = np.asarray(n_parameters, dtype=float)
    observations = np.asarray(n_observations, dtype=float)
    ratio = parameters / observations
    if hurvich_tsai:
        factor = observations / (observations - parameters - 1.0)
        aic = -2.0 * likelihood + 2.0 * ratio * factor
        aic = np.where(factor <= 0.0, np.nan, aic)
    else:
        aic = -2.0 * likelihood + 2.0 * ratio
    bic = -2.0 * likelihood + ratio * np.log(observations)
    hqc = -2.0 * likelihood + 2.0 * ratio * np.log(np.log(observations))
    return aic, bic, hqc


@dataclass(frozen=True)
class VARInformationCriteriaResult:
    """AIC, BIC, and HQC curves and their minimizing VAR orders."""

    p_aic: int
    p_bic: int
    p_hqc: int
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
        # Preserve constructor arguments exactly for sklearn BaseEstimator.
        self.orders = orders
        self.solver = solver
        self.hurvich_tsai = hurvich_tsai
        self.device = device
        self.dtype = dtype
        self.refit = refit

    def fit(self, X: np.ndarray | torch.Tensor, y=None):
        """Fit every candidate order and compute AIC, BIC, and HQC curves.

        Parameters
        ----------
        X
            Observations with shape ``(time, variables)`` or
            ``(batch, time, variables)``. Batched trajectories are fitted
            without constructing transitions across trajectory boundaries.
        y
            Unused scikit-learn compatibility target.

        Returns
        -------
        VAROrderSelectionIC
            Fitted selector with criterion curves and minimizing orders.
        """
        del y
        orders = tuple(int(value) for value in self.orders)
        if not orders or min(orders) < 1:
            raise ValueError("orders must be positive")
        data = torch.as_tensor(X)
        if data.ndim == 2:
            data = data.unsqueeze(0)
        if data.ndim != 3:
            raise ValueError("X must have shape (time,n) or (trials,time,n)")
        n_trials, n_times, n_variables = data.shape
        loglik = []
        n_parameters = []
        n_observations = []
        fitted = {}
        for order in orders:
            estimator = VAR(
                order=order,
                solver=self.solver,
                covariance="unbiased",
                mode="pooled" if n_trials > 1 else "independent",
                device=self.device,
                dtype=self.dtype,
                stability="ignore",
            ).fit(data)
            fitted[order] = estimator
            covariance = estimator.noise_covariance_[0]
            # For a Gaussian innovation model, E[e' Sigma^-1 e] = n_variables;
            # this gives the maximized per-observation log-likelihood used by
            # the MVGC information-criterion convention.
            loglik.append(
                -0.5
                * (
                    n_variables * np.log(2.0 * np.pi)
                    + float(spd_logdet(covariance))
                    + n_variables
                )
            )
            n_parameters.append(order * n_variables * n_variables)
            n_observations.append(n_trials * (n_times - order))
        likelihood = np.asarray(loglik)
        aic, bic, hqc = _information_criteria(
            likelihood,
            np.asarray(n_parameters),
            np.asarray(n_observations),
            hurvich_tsai=self.hurvich_tsai,
        )
        result = VARInformationCriteriaResult(
            orders[int(np.nanargmin(aic))],
            orders[int(np.nanargmin(bic))],
            orders[int(np.nanargmin(hqc))],
            aic,
            bic,
            hqc,
            likelihood,
            orders,
        )
        self.result_ = result
        self.p_aic_ = result.p_aic
        self.p_bic_ = result.p_bic
        self.p_hqc_ = result.p_hqc
        self.aic_ = aic
        self.bic_ = bic
        self.hqc_ = hqc
        self.loglik_ = likelihood
        if self.refit is not None:
            key = self.refit.lower()
            if key not in {"aic", "bic", "hqc"}:
                raise ValueError("refit must be None, 'aic', 'bic' or 'hqc'")
            self.best_order_ = getattr(result, f"p_{key}")
            self.best_estimator_ = fitted[self.best_order_]
        return self


@dataclass(frozen=True)
class VAROrderScore:
    """Held-out and information-criterion diagnostics for one VAR order."""

    order: int
    mean_score: float
    standard_error: float
    fold_scores: tuple[float, ...]
    failed_folds: int
    mean_train_aic: float
    mean_train_bic: float
    mean_train_hqc: float
    fold_train_aic: tuple[float, ...]
    fold_train_bic: tuple[float, ...]
    fold_train_hqc: tuple[float, ...]


@dataclass(frozen=True)
class VAROrderSearchResult:
    """Immutable summary returned by temporal VAR order search."""

    best_order: int
    scores: tuple[VAROrderScore, ...]
    scoring: str
    selection_rule: str

    def as_records(self):
        """Return records suitable for tabular serialization."""
        return [score.__dict__.copy() for score in self.scores]


class VAROrderSearchCV(_TemporalOrderSearchCV):
    """Select one global VAR lag order by temporal cross-validation.

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

    def _begin_diagnostics(self, n_orders, n_folds):
        """Allocate training-fold AIC, BIC, and HQC arrays."""
        self._aic = np.full((n_orders, n_folds), np.nan)
        self._bic = np.full((n_orders, n_folds), np.nan)
        self._hqc = np.full((n_orders, n_folds), np.nan)

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
        """Return held-out RMSE or Gaussian NLL."""
        if self.scoring == "rmse":
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
        return float(values.mean())

    def _fold_diagnostics(self, workspace, orders, fold_index):
        """Store fold and candidate indices needed by IC diagnostics."""
        workspace["fold_index"] = fold_index
        workspace["orders"] = orders

    def _evaluate_candidate(self, workspace, order, fold):
        """Fit and score one candidate and record its training IC values."""
        # Every candidate is fitted only on the chronological training prefix;
        # the held-out block contributes only to predictive loss.
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
        loglik = -0.5 * (
            n_variables * np.log(2.0 * np.pi)
            + float(logdet.mean())
            + n_variables
        )
        multiplier = n_trials if self.mode == "independent" else 1
        aic, bic, hqc = _information_criteria(
            loglik,
            multiplier * order * n_variables * n_variables,
            n_trials * (n_times - order),
            hurvich_tsai=self.hurvich_tsai,
        )
        self._aic[order_index, fold_index] = float(aic)
        self._bic[order_index, fold_index] = float(bic)
        self._hqc[order_index, fold_index] = float(hqc)

    def _finalize_diagnostics(self, orders):
        """Expose fold-level and mean IC diagnostics as fitted attributes."""
        del orders
        self.train_aic_ = self._aic
        self.train_bic_ = self._bic
        self.train_hqc_ = self._hqc
        self.mean_train_aic_ = np.nanmean(self._aic, axis=1)
        self.mean_train_bic_ = np.nanmean(self._bic, axis=1)
        self.mean_train_hqc_ = np.nanmean(self._hqc, axis=1)

    def _refit_best(self, data, order):
        """Refit the selected VAR order on all observations."""
        return self._fit_var(data, order)

    def fit(self, X, y=None):
        """Evaluate every candidate VAR order on every temporal fold."""
        del y
        summary = self._run_temporal_search(X)
        scores = tuple(
            VAROrderScore(
                order=order,
                mean_score=float(summary.mean_scores[index]),
                standard_error=float(summary.standard_errors[index]),
                fold_scores=tuple(summary.fold_scores[index]),
                failed_folds=int(summary.failed_folds[index]),
                mean_train_aic=float(self.mean_train_aic_[index]),
                mean_train_bic=float(self.mean_train_bic_[index]),
                mean_train_hqc=float(self.mean_train_hqc_[index]),
                fold_train_aic=tuple(self.train_aic_[index]),
                fold_train_bic=tuple(self.train_bic_[index]),
                fold_train_hqc=tuple(self.train_hqc_[index]),
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
