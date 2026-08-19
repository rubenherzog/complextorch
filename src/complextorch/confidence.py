r"""Public confidence-interval API for model-derived ComplexTorch measures.

The API delegates resampling and fixed-complexity refit mechanics to the private
:mod:`complextorch._resampling` engine, then evaluates the configured analytical
measure registry once on the original fitted model and once on the shared
batched resampling ensemble. VAR, general state-space, and innovations-form
estimators share the same observable-process bootstrap principle. No measure
triggers a second bootstrap.

References
----------
- Beda, A., Simpson, D. M., and Faes, L. (2017). Estimation of confidence
  limits for descriptive indexes derived from autoregressive analysis of time
  series. *PLoS ONE*, 12(10), e0186694.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from ._resampling import (
    ConfidenceIntervalResult,
    MeasureInterval,
    ResamplingMethod,
    _batched_ols_refit,
    _build_ensemble,
    _draw_innovations,
    _ensemble_config,
    _normalise_trials,
    _prediction_residuals,
    _simulate_resamples,
    _stable_resample_mask,
)
from ._state_space_resampling import (
    _fitted_canonical_system,
    _normalise_state_space_trials,
    _refit_state_space_resamples,
    _select_state_space_ensemble,
    _simulate_state_space_resamples,
    _state_space_prediction_innovations,
    _state_space_stable_mask,
)
from .inference_registry import InferenceMeasureConfig, evaluate_resampling_measures
from .representations import build_var_system
from .state_space import LarimoreStateSpace, LinearGaussianEM, N4SID
from .transformations import as_innovations_state_space
from .var import VAR


def _flatten_tensor_measures(value: Any, prefix: str = "") -> dict[str, torch.Tensor]:
    """Flatten nested tensor-valued measure trees while skipping metadata."""
    if torch.is_tensor(value):
        return {prefix: value} if prefix else {}
    output: dict[str, torch.Tensor] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {"context", "available", "not_available", "model_type"}:
                continue
            name = f"{prefix}.{key}" if prefix else str(key)
            output.update(_flatten_tensor_measures(child, name))
    return output


def _resolve_measure_names(
    available: Mapping[str, torch.Tensor],
    requested: str | Sequence[str],
) -> tuple[str, ...]:
    """Resolve full paths or unique leaf aliases against the computed registry."""
    if requested == "all_compatible":
        return tuple(sorted(available))
    raw = (requested,) if isinstance(requested, str) else tuple(requested)
    resolved: list[str] = []
    for name in raw:
        if name in available:
            resolved.append(name)
            continue
        matches = [path for path in available if path.rsplit(".", 1)[-1] == name]
        if len(matches) == 1:
            resolved.append(matches[0])
        elif not matches:
            raise ValueError(f"unknown or unavailable measure: {name}")
        else:
            raise ValueError(f"ambiguous measure name {name!r}; use one of {matches}")
    return tuple(dict.fromkeys(resolved))


def measure_confidence_intervals(
    X: np.ndarray | torch.Tensor,
    measures: str | Sequence[str] = "all_compatible",
    *,
    var: VAR | None = None,
    estimator: VAR | N4SID | LarimoreStateSpace | LinearGaussianEM | None = None,
    config: InferenceMeasureConfig | None = None,
    method: ResamplingMethod = "residual_bootstrap",
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
    return_samples: bool = False,
) -> ConfidenceIntervalResult:
    r"""Estimate percentile confidence intervals from one shared model ensemble.

    Model complexity is held fixed. Synthetic datasets are generated either by
    residual-vector bootstrap or Gaussian parametric Monte Carlo and refitted
    with the same estimator family. State-space systems are simulated through
    their exact steady-state innovations representation. Every configured
    analytical measure is then evaluated on the same valid model ensemble.

    Parameters
    ----------
    X
        Observations with shape ``(time, variables)`` or
        ``(batch, time, variables)``.
    measures
        ``"all_compatible"``, one flattened registry path, one unique leaf
        alias, or a sequence of names. Selection happens after the shared
        measure tree is evaluated and therefore never changes resampling cost.
    var
        Backward-compatible alias for a fixed-order unregularized OLS VAR
        estimator. Mutually exclusive with ``estimator``.
    estimator
        Fixed-complexity VAR or state-space estimator. State-space model
        selection is never repeated inside resampling. ``LinearGaussianEM``
        uses its fitted original system to initialize every bootstrap refit.
    config
        Inference measure configuration. Its ``primary.macro_projection`` is a
        fixed DI/DD projection or fixed batch of projections; optimization is
        never performed during resampling.
    method
        ``"residual_bootstrap"`` or ``"parametric"``.
    n_resamples
        Number of synthetic refits requested.
    confidence
        Central percentile confidence level in ``(0, 1)``.
    seed
        Optional local Torch random seed.
    return_samples
        Retain bootstrap measure samples when true.

    Returns
    -------
    ConfidenceIntervalResult
        Point estimates, percentile limits, and resampling diagnostics.

    Notes
    -----
    In ``mode="pooled"`` no lag row crosses a trajectory boundary. In
    ``mode="independent"`` each accepted bootstrap replicate requires every
    trajectory-specific fitted system to be stable so all measures retain a
    common resampling axis.
    """
    if n_resamples < 2:
        raise ValueError("n_resamples must be at least two")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    if var is not None and estimator is not None:
        raise ValueError("pass either var or estimator, not both")
    fitted_estimator = copy.deepcopy(
        estimator
        if estimator is not None
        else (VAR(order=1, mode="pooled") if var is None else var)
    )
    is_var = isinstance(fitted_estimator, VAR)
    if is_var:
        if fitted_estimator.alpha != 0:
            raise ValueError("ARres inference currently requires an unregularized VAR")
        if fitted_estimator.solver == "lwr":
            raise ValueError(
                "ARres inference currently supports OLS VAR estimators, "
                "not solver='lwr'"
            )
        trials = _normalise_trials(fitted_estimator, X)
        fitted_estimator.fit(trials)
        residuals = _prediction_residuals(fitted_estimator, trials)
        covariance_for_draw = fitted_estimator.noise_covariance_
        original_system = build_var_system(
            fitted_estimator.coef_, fitted_estimator.noise_covariance_
        )
    else:
        if not isinstance(
            fitted_estimator, (N4SID, LarimoreStateSpace, LinearGaussianEM)
        ):
            raise TypeError(
                "estimator must be VAR, N4SID, LarimoreStateSpace, "
                "or LinearGaussianEM"
            )
        trials = _normalise_state_space_trials(fitted_estimator, X)
        # State-space estimators have no intercept; use their zero-mean convention.
        trials = trials - trials.mean(dim=1, keepdim=True)
        fitted_estimator.fit(trials)
        original_system = _fitted_canonical_system(fitted_estimator)
        residuals = _state_space_prediction_innovations(original_system, trials)
        covariance_for_draw = as_innovations_state_space(
            original_system
        ).innovation_covariance

    generator = torch.Generator(device=trials.device)
    if seed is not None:
        generator.manual_seed(seed)
    innovations = _draw_innovations(
        residuals,
        covariance_for_draw,
        mode=fitted_estimator.mode,
        method=method,
        n_resamples=n_resamples,
        generator=generator,
    )
    if is_var:
        synthetic = _simulate_resamples(trials, fitted_estimator, innovations)
        coefficients, covariance = _batched_ols_refit(synthetic, fitted_estimator)
        valid = _stable_resample_mask(
            coefficients,
            n_resamples=n_resamples,
            n_trials=trials.shape[0],
            mode=fitted_estimator.mode,
        )
        ensemble = _build_ensemble(
            coefficients,
            covariance,
            valid,
            n_resamples=n_resamples,
            n_trials=trials.shape[0],
            mode=fitted_estimator.mode,
        )
    else:
        synthetic = _simulate_state_space_resamples(original_system, innovations)
        refitted = _refit_state_space_resamples(
            synthetic, fitted_estimator, original_system
        )
        valid = _state_space_stable_mask(
            refitted,
            n_resamples=n_resamples,
            n_trials=trials.shape[0],
            mode=fitted_estimator.mode,
        )
        ensemble = _select_state_space_ensemble(
            refitted,
            valid,
            n_resamples=n_resamples,
            n_trials=trials.shape[0],
            mode=fitted_estimator.mode,
        )
    n_valid = int(valid.sum().item())
    if n_valid < 2:
        raise RuntimeError("fewer than two stable resampled models remain")

    measure_config = InferenceMeasureConfig() if config is None else config
    ensemble_primary = _ensemble_config(
        measure_config.primary,
        n_valid=n_valid,
        n_trials=trials.shape[0],
        mode=fitted_estimator.mode,
    )
    ensemble_config = InferenceMeasureConfig(
        primary=ensemble_primary,
        oir_groups=measure_config.oir_groups,
        delta_oir_target_group=measure_config.delta_oir_target_group,
        hop_sources=measure_config.hop_sources,
        hop_target=measure_config.hop_target,
        half_open=measure_config.half_open,
    )
    original_flat = _flatten_tensor_measures(
        evaluate_resampling_measures(original_system, measure_config)
    )
    ensemble_flat = _flatten_tensor_measures(
        evaluate_resampling_measures(ensemble, ensemble_config)
    )
    names = _resolve_measure_names(original_flat, measures)
    missing = [name for name in names if name not in ensemble_flat]
    if missing:
        raise RuntimeError(f"measures unavailable on bootstrap ensemble: {missing}")

    alpha = (1.0 - confidence) / 2.0
    intervals: dict[str, MeasureInterval] = {}
    for name in names:
        estimate = original_flat[name]
        samples = ensemble_flat[name]
        if fitted_estimator.mode == "pooled":
            if samples.shape[0] != n_valid:
                raise RuntimeError(
                    f"measure {name} does not preserve the model batch axis"
                )
            estimate = estimate[0] if estimate.shape[:1] == (1,) else estimate
            sample_view = samples
        else:
            expected = n_valid * trials.shape[0]
            if samples.shape[0] != expected:
                raise RuntimeError(
                    f"measure {name} does not preserve the model batch axis"
                )
            sample_view = samples.reshape(
                n_valid, trials.shape[0], *samples.shape[1:]
            )
        intervals[name] = MeasureInterval(
            estimate=estimate,
            lower=torch.quantile(sample_view, alpha, dim=0),
            upper=torch.quantile(sample_view, 1.0 - alpha, dim=0),
            samples=sample_view if return_samples else None,
            n_valid=n_valid,
        )
    return ConfidenceIntervalResult(
        intervals=intervals,
        method=method,
        confidence=confidence,
        n_resamples=n_resamples,
        n_valid=n_valid,
        n_failed=n_resamples - n_valid,
        seed=seed,
        var_order=fitted_estimator.order if is_var else None,
        fit_mode=fitted_estimator.mode,
        estimator_type=type(fitted_estimator).__name__,
    )


__all__ = ["InferenceMeasureConfig", "measure_confidence_intervals"]
