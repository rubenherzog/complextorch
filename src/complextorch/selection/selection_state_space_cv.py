"""Estimator-agnostic temporal cross-validation for state-space order search.

This module generalizes the validated Larimore state-space CV search so the
same leakage-safe temporal splitting, prediction, scoring, selection, and batch
semantics can be reused by other fixed-order state-space estimators. Larimore
retains its optimized shared-decomposition path; N4SID and compatible custom
estimators are fitted independently at each candidate latent dimension.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Iterable, Literal

import numpy as np
import torch
from sklearn.base import BaseEstimator, clone

from ..control import InnovationsStateSpace, innovations_form
from ..state_space import LarimoreStateSpace, N4SID, _normalise_ss_observations
from ._temporal import EpochTimeSeriesSplit
from .selection_state_space import (
    StateSpaceOrderSearchCV as _LarimoreStateSpaceOrderSearchCV,
)


class StateSpaceOrderSearchCV(_LarimoreStateSpaceOrderSearchCV):
    r"""Select latent state dimension by temporal cross-validation.

    ``method="larimore"`` preserves the optimized Larimore CVA path in which
    each fold computes one maximum-rank decomposition and candidate dimensions
    truncate that shared basis. ``method="n4sid"`` fits fixed-order
    :class:`~complextorch.N4SID` candidates under the same temporal folds,
    held-out prediction logic, scoring, and selection rule.

    A scikit-learn-compatible fixed-order estimator prototype may instead be
    supplied with ``estimator=``. It must expose an ``n_states`` parameter;
    each candidate is a clone with only ``n_states`` (and ``mode`` when the
    estimator supports it) changed by the search.

    Batch semantics match :class:`~complextorch.VAROrderSearchCV`:
    ``mode="pooled"`` selects one common latent dimension across independent
    trajectories, while batched ``mode="independent"`` preserves one CV curve
    and one selected dimension per trajectory. No temporal relationship is ever
    formed across trajectory boundaries.

    General state-space estimates such as N4SID are converted to their
    steady-state innovations form for held-out prediction, so all backends are
    scored under the same predictive contract.
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
        method: Literal["larimore", "n4sid"] = "larimore",
        estimator: BaseEstimator | None = None,
    ):
        """Initialize estimator-agnostic temporal state-space order search."""
        super().__init__(
            orders,
            past_horizon,
            future_horizon=future_horizon,
            cv=cv,
            scoring=scoring,
            selection_rule=selection_rule,
            ridge=ridge,
            covariance=covariance,
            mode=mode,
            device=device,
            dtype=dtype,
            prediction_mode=prediction_mode,
            gap_mode=gap_mode,
            refit=refit,
            bauer_diagnostics=bauer_diagnostics,
        )
        self.method = method
        self.estimator = estimator

    def _validate_common_settings(self):
        """Validate shared temporal settings and backend configuration."""
        orders = super()._validate_common_settings()
        if self.method not in {"larimore", "n4sid"}:
            raise ValueError("method must be 'larimore' or 'n4sid'")
        if self.estimator is not None and not isinstance(
            self.estimator, BaseEstimator
        ):
            raise TypeError("estimator must be a scikit-learn compatible estimator")
        if self.estimator is not None and self.method != "larimore":
            raise ValueError("specify either estimator or method, not both")
        if self.estimator is not None:
            parameters = self.estimator.get_params(deep=False)
            if "n_states" not in parameters:
                raise ValueError(
                    "estimator must expose an 'n_states' parameter for order search"
                )
        return orders

    def _method_name(self):
        """Return the effective identification backend name."""
        if isinstance(self.estimator, LarimoreStateSpace):
            return "larimore"
        if isinstance(self.estimator, N4SID):
            return "n4sid"
        if self.estimator is not None:
            return type(self.estimator).__name__.lower()
        return self.method

    def _prepare_fold(self, data, fold, orders):
        """Center one fold and prepare backend-specific training state."""
        if self._method_name() == "larimore":
            return super()._prepare_fold(data, fold, orders)
        values, _ = _normalise_ss_observations(
            data, device=self.device, dtype=self.dtype
        )
        training_mean = values[:, : fold.train_stop].mean(dim=1, keepdim=True)
        centered = values - training_mean
        return {
            "centered": centered,
            "training": centered[:, : fold.train_stop],
            "orders": orders,
        }

    def _fold_diagnostics(self, workspace, orders, fold_index):
        """Expose Bauer diagnostics only for the Larimore CVA backend."""
        if self._method_name() != "larimore":
            return
        super()._fold_diagnostics(workspace, orders, fold_index)

    def _new_estimator(self, order, *, mode=None):
        """Construct one unfitted fixed-order estimator for a candidate."""
        fit_mode = self.mode if mode is None else mode
        if self.estimator is not None:
            estimator = clone(self.estimator)
            parameters = estimator.get_params(deep=False)
            updates = {"n_states": order}
            if "mode" in parameters:
                updates["mode"] = fit_mode
            return estimator.set_params(**updates)
        if self.method == "n4sid":
            return N4SID(
                n_states=order,
                block_rows=self.past_horizon,
                ridge=self.ridge,
                mode=fit_mode,
                device=self.device,
                dtype=self.dtype,
            )
        return LarimoreStateSpace(
            n_states=order,
            past_horizon=self.past_horizon,
            future_horizon=self.future_horizon,
            ridge=self.ridge,
            covariance=self.covariance,
            mode=fit_mode,
            device=self.device,
            dtype=self.dtype,
        )

    def _fit_candidate(self, workspace, order):
        """Fit or truncate one candidate using the selected backend."""
        if self._method_name() == "larimore" and "decomposition" in workspace:
            return workspace["decomposition"].fit_order(
                order, covariance=self.covariance
            )
        return self._new_estimator(order).fit(workspace["training"])

    @staticmethod
    def _innovations_system(estimator):
        """Normalize fitted estimator output to steady-state innovations form."""
        system = estimator.system_
        if isinstance(system, InnovationsStateSpace) or hasattr(system, "gain"):
            return system
        form = innovations_form(system)
        return InnovationsStateSpace(
            system.transition,
            system.observation,
            form.gain,
            form.covariance,
        )

    def _innovation_errors(self, estimator, observations, fold):
        """Compute held-out innovations under the common prediction contract."""
        wrapper = SimpleNamespace(system_=self._innovations_system(estimator))
        return super()._innovation_errors(wrapper, observations, fold)

    def _fit_and_score_fold(self, data, order, fold):
        """Compatibility hook fitting and scoring one candidate on one fold."""
        orders = tuple(sorted({int(value) for value in self.orders}))
        workspace = self._prepare_fold(data, fold, orders)
        estimator = self._fit_candidate(workspace, order)
        errors = self._innovation_errors(estimator, workspace["centered"], fold)
        system = self._innovations_system(estimator)
        return self._score(errors, system.innovation_covariance)

    def _evaluate_candidate(self, workspace, order, fold):
        """Fit one candidate and return its held-out temporal loss."""
        if type(self)._fit_and_score_fold is not StateSpaceOrderSearchCV._fit_and_score_fold:
            return self._fit_and_score_fold(workspace["centered"], order, fold)
        estimator = self._fit_candidate(workspace, order)
        errors = self._innovation_errors(estimator, workspace["centered"], fold)
        system = self._innovations_system(estimator)
        return self._score(errors, system.innovation_covariance)

    def _fit_full(self, data, order, *, mode=None):
        """Fit one selected fixed-order model on the complete observations."""
        return self._new_estimator(order, mode=mode).fit(data)

    def _refit_best(self, data, order):
        """Refit scalar or per-trajectory selected latent dimensions."""
        selected = np.asarray(order)
        if selected.ndim == 0:
            return self._fit_full(data, int(selected))
        if selected.shape != (data.shape[0],):
            raise ValueError("selected orders must match the batch dimension")
        return tuple(
            self._fit_full(
                data[index : index + 1],
                int(value),
                mode="independent",
            )
            for index, value in enumerate(selected)
        )

    def fit(self, X, y=None):
        """Run temporal CV and expose the effective backend as ``method_``."""
        result = super().fit(X, y=y)
        self.method_ = self._method_name()
        return result


__all__ = ["StateSpaceOrderSearchCV"]
