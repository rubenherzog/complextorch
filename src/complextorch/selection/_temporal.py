"""Shared temporal cross-validation engine for model-order searches.

The engine treats model order as a global hyperparameter: every candidate is
fitted on every training fold, evaluated only on its held-out block, aggregated
across folds, and optionally refitted on all observations. Model-specific
subclasses provide fold preparation, candidate fitting/scoring, diagnostics,
and final refitting.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Literal

import numpy as np
import torch
from sklearn.base import BaseEstimator



@dataclass(frozen=True)
class TemporalFold:
    """Indices delimiting one expanding-window temporal-validation fold.

    Notes
    -----
    Public fitted attributes use the trailing-underscore convention.
    """
    train_stop: int
    test_start: int
    test_stop: int


class EpochTimeSeriesSplit:
    """Generate leakage-safe expanding-window splits for ordered observations.

    Notes
    -----
    Public fitted attributes use the trailing-underscore convention.
    """
    def __init__(
        self,
        n_splits: int = 5,
        *,
        test_size: int | None = None,
        min_train_size: int | None = None,
        gap: int = 0,
    ):
        """Initialize the estimator or result container.

        Parameters
        ----------
        n_splits
            Number of temporal validation folds.
        test_size
            Number of held-out samples in each fold.
        min_train_size
            Minimum number of samples in the first training window.
        gap
            Number of samples omitted between training and test windows.

        Notes
        -----
        Batch dimensions are preserved unless explicitly documented otherwise.
        The implementation validates dimensional and positive-definiteness
        requirements before executing the numerical core.
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.min_train_size = min_train_size
        self.gap = gap

    def split(self, n_times: int, *, min_order: int = 1):
        """Split.

        Parameters
        ----------
        n_times
            Number of time samples.
        min_order
            Largest lag that must fit inside every training window.

        Returns
        -------
        object
            Iterator of :class:`TemporalFold` objects in chronological order.

        Notes
        -----
        Batch dimensions are preserved unless explicitly documented otherwise.
        The implementation validates dimensional and positive-definiteness
        requirements before executing the numerical core.
        """
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


PredictionMode = Literal["rolling", "recursive"]
GapMode = Literal["warmup", "embargo"]
SelectionRule = Literal["best", "one_se"]
Scoring = Literal["nll", "rmse"]


@dataclass(frozen=True)
class _TemporalSearchSummary:
    """Model-agnostic numerical output of temporal order search."""

    orders: tuple[int, ...]
    folds: tuple[TemporalFold, ...]
    fold_scores: np.ndarray
    mean_scores: np.ndarray
    standard_errors: np.ndarray
    failed_folds: np.ndarray
    best_order: int
    failure_messages: dict[tuple[int, int], str]


class _TemporalOrderSearchCV(BaseEstimator):
    """Internal base implementing the common temporal order-search loop."""

    _expected_errors = (RuntimeError, ValueError, torch.linalg.LinAlgError)

    def _set_temporal_search_parameters(
        self,
        *,
        orders,
        cv,
        scoring: Scoring,
        selection_rule: SelectionRule,
        prediction_mode: PredictionMode,
        gap_mode: GapMode,
        refit: bool,
    ) -> None:
        """Store common constructor arguments without hidden mutation."""

        self.orders = tuple(int(order) for order in orders)
        self.cv = cv or EpochTimeSeriesSplit()
        self.scoring = scoring
        self.selection_rule = selection_rule
        self.prediction_mode = prediction_mode
        self.gap_mode = gap_mode
        self.refit = refit

    @staticmethod
    def _normalise_observations(X: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Return finite observations with shape ``(batch, time, variables)``."""

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

    def _validate_common_settings(self) -> tuple[int, ...]:
        """Validate common public settings and return sorted unique orders."""

        orders = tuple(sorted(set(self.orders)))
        if not orders or orders[0] < 1:
            raise ValueError("orders must contain positive integers")
        if self.scoring not in {"nll", "rmse"}:
            raise ValueError("scoring must be 'nll' or 'rmse'")
        if self.selection_rule not in {"best", "one_se"}:
            raise ValueError("selection_rule must be 'best' or 'one_se'")
        if self.prediction_mode not in {"rolling", "recursive"}:
            raise ValueError("prediction_mode must be 'rolling' or 'recursive'")
        if self.gap_mode not in {"warmup", "embargo"}:
            raise ValueError("gap_mode must be 'warmup' or 'embargo'")
        return orders

    def _minimum_training_size(self, orders: tuple[int, ...]) -> int:
        """Return model-specific minimum first-fold training length."""

        return max(orders)

    def _prepare_fold(
        self,
        data: torch.Tensor,
        fold: TemporalFold,
        orders: tuple[int, ...],
    ):
        """Build reusable training-fold state; implemented by subclasses."""

        raise NotImplementedError

    def _evaluate_candidate(self, workspace, order: int, fold: TemporalFold) -> float:
        """Fit and score one candidate; implemented by subclasses."""
        # Fit each candidate on every training window and aggregate held-out predictive loss across folds.

        raise NotImplementedError

    def _fold_diagnostics(
        self,
        workspace,
        orders: tuple[int, ...],
        fold_index: int,
    ) -> None:
        """Collect model-specific training diagnostics."""

    def _finalize_diagnostics(self, orders: tuple[int, ...]) -> None:
        """Expose model-specific diagnostic arrays after search."""

    def _refit_best(self, data: torch.Tensor, order: int):
        """Fit the selected candidate on all observations."""

        raise NotImplementedError

    @staticmethod
    def _aggregate_scores(fold_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Aggregate finite fold scores and count failures per candidate."""

        n_orders = fold_scores.shape[0]
        means = np.full(n_orders, np.inf, dtype=float)
        standard_errors = np.zeros(n_orders, dtype=float)
        failed = np.zeros(n_orders, dtype=int)
        for index in range(n_orders):
            finite = fold_scores[index, np.isfinite(fold_scores[index])]
            failed[index] = fold_scores.shape[1] - finite.size
            if finite.size:
                means[index] = float(finite.mean())
            if finite.size > 1:
                standard_errors[index] = float(
                    finite.std(ddof=1) / sqrt(finite.size)
                )
        return means, standard_errors, failed

    def _select_order(
        self,
        orders: tuple[int, ...],
        means: np.ndarray,
        standard_errors: np.ndarray,
    ) -> int:
        """Select the minimum-loss or one-standard-error candidate."""

        finite = np.flatnonzero(np.isfinite(means))
        if finite.size == 0:
            raise RuntimeError("all candidate orders failed")
        optimum_index = int(finite[np.argmin(means[finite])])
        if self.selection_rule == "best":
            return orders[optimum_index]
        threshold = means[optimum_index] + standard_errors[optimum_index]
        return min(
            orders[index]
            for index in finite
            if means[index] <= threshold
        )

    def _run_temporal_search(self, X: np.ndarray | torch.Tensor) -> _TemporalSearchSummary:
        """Execute the common fit-evaluate-aggregate-select-refit workflow."""

        orders = self._validate_common_settings()
        data = self._normalise_observations(X)
        folds = tuple(
            self.cv.split(
                data.shape[1],
                min_order=self._minimum_training_size(orders),
            )
        )
        if not folds:
            raise ValueError("cv produced no folds")

        self._begin_diagnostics(len(orders), len(folds))
        fold_scores = np.full((len(orders), len(folds)), np.inf, dtype=float)
        failures: dict[tuple[int, int], str] = {}

        for fold_index, fold in enumerate(folds):
            try:
                workspace = self._prepare_fold(data, fold, orders)
                self._fold_diagnostics(workspace, orders, fold_index)
            except self._expected_errors as error:
                for order in orders:
                    failures[(order, fold_index)] = f"fold preparation: {error}"
                continue

            for order_index, order in enumerate(orders):
                try:
                    fold_scores[order_index, fold_index] = self._evaluate_candidate(
                        workspace, order, fold
                    )
                except self._expected_errors as error:
                    failures[(order, fold_index)] = str(error)

        means, standard_errors, failed = self._aggregate_scores(fold_scores)
        best_order = self._select_order(orders, means, standard_errors)

        self.orders_ = np.asarray(orders, dtype=int)
        self.folds_ = folds
        self.fold_scores_ = fold_scores
        self.mean_test_scores_ = means
        self.standard_error_ = standard_errors
        self.failed_folds_ = failed
        self.best_order_ = best_order
        self.failure_messages_ = failures
        self._finalize_diagnostics(orders)
        if self.refit:
            self.best_estimator_ = self._refit_best(data, best_order)

        return _TemporalSearchSummary(
            orders=orders,
            folds=folds,
            fold_scores=fold_scores,
            mean_scores=means,
            standard_errors=standard_errors,
            failed_folds=failed,
            best_order=best_order,
            failure_messages=failures,
        )

    def _begin_diagnostics(self, n_orders: int, n_folds: int) -> None:
        """Initialize model-specific diagnostic storage."""
