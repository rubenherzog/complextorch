"""Temporal validation and MVGC-compatible information-criterion order selection."""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Literal

import numpy as np
import torch
from sklearn.base import BaseEstimator

from ._typing import ArrayLike
from .linalg import stable_cholesky, spd_logdet
from .var import VAR


@dataclass(frozen=True)
class TemporalFold:
    train_stop: int
    test_start: int
    test_stop: int


class EpochTimeSeriesSplit:
    def __init__(
        self,
        n_splits: int = 5,
        *,
        test_size: int | None = None,
        min_train_size: int | None = None,
        gap: int = 0,
    ):
        self.n_splits = n_splits
        self.test_size = test_size
        self.min_train_size = min_train_size
        self.gap = gap

    def split(self, n_times: int, *, min_order: int = 1):
        if self.n_splits < 1 or self.gap < 0:
            raise ValueError("invalid split settings")
        test_size = self.test_size or max(1, n_times // (self.n_splits + 2))
        min_train = self.min_train_size or max(
            min_order + 5,
            n_times - self.n_splits * test_size - self.gap,
        )
        if min_train + self.gap + self.n_splits * test_size > n_times:
            raise ValueError("requested folds do not fit")
        for fold in range(self.n_splits):
            train_stop = min_train + fold * test_size
            test_start = train_stop + self.gap
            yield TemporalFold(train_stop, test_start, test_start + test_size)


def _information_criteria(
    loglik: float | np.ndarray,
    n_parameters: int | np.ndarray,
    n_observations: int | np.ndarray,
    *,
    hurvich_tsai: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return MVGC2-compatible per-observation AIC, BIC and HQC."""
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


def _estimator_information_criteria(
    estimator: VAR,
    *,
    n_trials: int,
    n_times: int,
    n_variables: int,
    order: int,
    hurvich_tsai: bool,
) -> tuple[float, float, float, float]:
    """Compute training-fold IC diagnostics from an already fitted VAR."""
    covariance = estimator.noise_covariance_
    logdet = spd_logdet(covariance)
    # Independent fits have one covariance per trial. Their average
    # per-observation log likelihood is the mean across trials. For pooled
    # fits there is only one covariance matrix.
    loglik = -0.5 * (
        n_variables * np.log(2.0 * np.pi)
        + float(logdet.mean())
        + n_variables
    )
    parameter_multiplier = n_trials if estimator.mode == "independent" else 1
    n_parameters = parameter_multiplier * order * n_variables * n_variables
    n_observations = n_trials * (n_times - order)
    aic, bic, hqc = _information_criteria(
        loglik,
        n_parameters,
        n_observations,
        hurvich_tsai=hurvich_tsai,
    )
    return loglik, float(aic), float(bic), float(hqc)


@dataclass(frozen=True)
class VAROrderScore:
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
    best_order: int
    scores: tuple[VAROrderScore, ...]
    scoring: str
    selection_rule: str

    def as_records(self):
        return [
            {
                "order": score.order,
                "mean_score": score.mean_score,
                "standard_error": score.standard_error,
                "fold_scores": score.fold_scores,
                "failed_folds": score.failed_folds,
                "mean_train_aic": score.mean_train_aic,
                "mean_train_bic": score.mean_train_bic,
                "mean_train_hqc": score.mean_train_hqc,
                "fold_train_aic": score.fold_train_aic,
                "fold_train_bic": score.fold_train_bic,
                "fold_train_hqc": score.fold_train_hqc,
            }
            for score in self.scores
        ]


class VAROrderSearchCV:
    """Temporal cross-validation with training-fold AIC/BIC/HQC diagnostics.

    Model selection is based only on the held-out ``scoring`` value and
    ``selection_rule``. AIC, BIC and HQC are calculated on each training fold
    using the same MVGC2-compatible formulas as :class:`VAROrderSelectionIC`.
    They are diagnostic outputs and never participate in selecting
    ``best_order_``.
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
        refit: bool = True,
        hurvich_tsai: bool = False,
    ):
        self.orders = tuple(int(value) for value in orders)
        self.cv = cv or EpochTimeSeriesSplit()
        self.scoring = scoring
        self.selection_rule = selection_rule
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.mode = mode
        self.solver = solver
        self.device = device
        self.dtype = dtype
        self.prediction_mode = prediction_mode
        self.refit = refit
        self.hurvich_tsai = hurvich_tsai

    @staticmethod
    def _normalise(x: ArrayLike):
        tensor = torch.as_tensor(x)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3:
            raise ValueError("X must have shape (time,n) or (batch,time,n)")
        return tensor

    @staticmethod
    def _forecast_nll(errors, covariance):
        if covariance.shape[0] == 1 and errors.shape[0] > 1:
            covariance = covariance.expand(errors.shape[0], -1, -1)
        chol, _ = stable_cholesky(covariance, jitter=1e-10)
        solved = torch.cholesky_solve(
            errors.unsqueeze(-1), chol[:, None]
        ).squeeze(-1)
        return 0.5 * (
            (errors * solved).sum(-1)
            + 2.0
            * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)[:, None]
            + errors.shape[-1] * np.log(2.0 * np.pi)
        )

    def _fit_and_score_fold(self, data, order, fold):
        training = data[:, : fold.train_stop]
        estimator = VAR(
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

        loglik, aic, bic, hqc = _estimator_information_criteria(
            estimator,
            n_trials=training.shape[0],
            n_times=training.shape[1],
            n_variables=training.shape[2],
            order=order,
            hurvich_tsai=self.hurvich_tsai,
        )

        if self.prediction_mode == "rolling":
            prediction = estimator.one_step_predictions(
                data[:, : fold.test_stop]
            )[:, fold.test_start - order : fold.test_stop - order]
        else:
            forecast = estimator.forecast(
                training, fold.test_stop - fold.train_stop
            )
            start = fold.test_start - fold.train_stop
            prediction = forecast[:, start : start + fold.test_stop - fold.test_start]
        errors = data[:, fold.test_start : fold.test_stop].to(prediction) - prediction
        if self.scoring == "rmse":
            score = float(torch.sqrt(torch.mean(errors.square())))
        elif self.scoring == "nll":
            score = float(
                self._forecast_nll(errors, estimator.noise_covariance_).mean()
            )
        else:
            raise ValueError("unknown scoring")
        return score, loglik, aic, bic, hqc

    @staticmethod
    def _finite_mean(values: list[float]) -> float:
        finite = np.asarray([value for value in values if np.isfinite(value)])
        return float(np.mean(finite)) if finite.size else float("nan")

    def fit(self, X: ArrayLike, y=None):
        del y
        if not self.orders or min(self.orders) < 1:
            raise ValueError("orders must be positive")
        data = self._normalise(X)
        folds = tuple(
            self.cv.split(data.shape[1], min_order=max(self.orders))
        )
        results = []
        for order in sorted(set(self.orders)):
            scores: list[float] = []
            aic_values: list[float] = []
            bic_values: list[float] = []
            hqc_values: list[float] = []
            failed = 0
            for fold in folds:
                try:
                    score, _, aic, bic, hqc = self._fit_and_score_fold(
                        data, order, fold
                    )
                    scores.append(score)
                    aic_values.append(aic)
                    bic_values.append(bic)
                    hqc_values.append(hqc)
                except (RuntimeError, ValueError, torch.linalg.LinAlgError):
                    scores.append(float("inf"))
                    aic_values.append(float("nan"))
                    bic_values.append(float("nan"))
                    hqc_values.append(float("nan"))
                    failed += 1
            finite_scores = np.asarray(
                [value for value in scores if np.isfinite(value)]
            )
            mean = (
                float(np.mean(finite_scores))
                if finite_scores.size
                else float("inf")
            )
            standard_error = (
                float(np.std(finite_scores, ddof=1) / sqrt(finite_scores.size))
                if finite_scores.size > 1
                else 0.0
            )
            results.append(
                VAROrderScore(
                    order=order,
                    mean_score=mean,
                    standard_error=standard_error,
                    fold_scores=tuple(scores),
                    failed_folds=failed,
                    mean_train_aic=self._finite_mean(aic_values),
                    mean_train_bic=self._finite_mean(bic_values),
                    mean_train_hqc=self._finite_mean(hqc_values),
                    fold_train_aic=tuple(aic_values),
                    fold_train_bic=tuple(bic_values),
                    fold_train_hqc=tuple(hqc_values),
                )
            )

        finite_results = [
            result for result in results if np.isfinite(result.mean_score)
        ]
        if not finite_results:
            raise RuntimeError("all candidate orders failed")
        optimum = min(finite_results, key=lambda result: result.mean_score)
        if self.selection_rule == "best":
            best = optimum.order
        elif self.selection_rule == "one_se":
            best = min(
                result.order
                for result in finite_results
                if result.mean_score
                <= optimum.mean_score + optimum.standard_error
            )
        else:
            raise ValueError("unknown selection rule")

        self.result_ = VAROrderSearchResult(
            best,
            tuple(results),
            self.scoring,
            self.selection_rule,
        )
        self.best_order_ = best
        self.cv_results_ = self.result_.as_records()
        self.train_aic_ = np.asarray(
            [result.fold_train_aic for result in results], dtype=float
        )
        self.train_bic_ = np.asarray(
            [result.fold_train_bic for result in results], dtype=float
        )
        self.train_hqc_ = np.asarray(
            [result.fold_train_hqc for result in results], dtype=float
        )
        self.mean_train_aic_ = np.asarray(
            [result.mean_train_aic for result in results], dtype=float
        )
        self.mean_train_bic_ = np.asarray(
            [result.mean_train_bic for result in results], dtype=float
        )
        self.mean_train_hqc_ = np.asarray(
            [result.mean_train_hqc for result in results], dtype=float
        )
        if self.refit:
            self.best_estimator_ = VAR(
                order=best,
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                mode=self.mode,
                solver=self.solver,
                device=self.device,
                dtype=self.dtype,
            ).fit(data)
        return self


@dataclass(frozen=True)
class VARInformationCriteriaResult:
    p_aic: int
    p_bic: int
    p_hqc: int
    aic: np.ndarray
    bic: np.ndarray
    hqc: np.ndarray
    loglik: np.ndarray
    orders: tuple[int, ...]


class VAROrderSelectionIC(BaseEstimator):
    """MVGC2-compatible in-sample AIC/BIC/HQC order selection."""

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
        self.orders = tuple(int(value) for value in orders)
        self.solver = solver
        self.hurvich_tsai = hurvich_tsai
        self.device = device
        self.dtype = dtype
        self.refit = refit

    def fit(self, X: ArrayLike, y=None):
        del y
        if not self.orders or min(self.orders) < 1:
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
        for order in self.orders:
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
            self.orders[int(np.nanargmin(aic))],
            self.orders[int(np.nanargmin(bic))],
            self.orders[int(np.nanargmin(hqc))],
            aic,
            bic,
            hqc,
            likelihood,
            self.orders,
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
