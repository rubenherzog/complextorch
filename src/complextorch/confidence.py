r"""Public confidence-interval API for model-derived ComplexTorch measures.

The API delegates all resampling and VAR-refit mechanics to
:mod:`complextorch.inference`, then evaluates the configured analytical measure
registry once on the original fitted model and once on the shared batched
resampling ensemble. No measure triggers a second bootstrap.

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

from .inference import (
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
from .inference_registry import InferenceMeasureConfig, evaluate_resampling_measures
from .representations import build_var_system
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
    config: InferenceMeasureConfig | None = None,
    method: ResamplingMethod = "residual_bootstrap",
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
    return_samples: bool = False,
) -> ConfidenceIntervalResult:
    r"""Estimate percentile confidence intervals from one shared VAR ensemble.

    The fitted VAR order is held fixed. Synthetic datasets are generated either
    by residual-vector bootstrap or Gaussian parametric Monte Carlo, refitted in
    batch, and filtered for stationarity. Every configured analytical measure is
    then evaluated on that same valid VAR ensemble.

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
        Fixed-order unregularized OLS VAR estimator. If omitted, uses
        ``VAR(order=1, mode="pooled")``.
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
    trajectory-specific VAR to be stable so all measures retain a common
    resampling axis.
    """
    if n_resamples < 2:
        raise ValueError("n_resamples must be at least two")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    estimator = copy.deepcopy(VAR(order=1, mode="pooled") if var is None else var)
    if estimator.alpha != 0:
        raise ValueError("ARres inference currently requires an unregularized VAR")
    if estimator.solver == "lwr":
        raise ValueError("ARres inference currently supports OLS VAR estimators, not solver='lwr'")

    trials = _normalise_trials(estimator, X)
    estimator.fit(trials)
    residuals = _prediction_residuals(estimator, trials)
    generator = torch.Generator(device=trials.device)
    if seed is not None:
        generator.manual_seed(seed)
    innovations = _draw_innovations(
        residuals,
        estimator.noise_covariance_,
        mode=estimator.mode,
        method=method,
        n_resamples=n_resamples,
        generator=generator,
    )
    synthetic = _simulate_resamples(trials, estimator, innovations)
    coefficients, covariance = _batched_ols_refit(synthetic, estimator)
    valid = _stable_resample_mask(
        coefficients,
        n_resamples=n_resamples,
        n_trials=trials.shape[0],
        mode=estimator.mode,
    )
    n_valid = int(valid.sum().item())
    if n_valid < 2:
        raise RuntimeError("fewer than two stable resampled VARs remain")
    ensemble = _build_ensemble(
        coefficients,
        covariance,
        valid,
        n_resamples=n_resamples,
        n_trials=trials.shape[0],
        mode=estimator.mode,
    )
    original_system = build_var_system(estimator.coef_, estimator.noise_covariance_)

    measure_config = InferenceMeasureConfig() if config is None else config
    ensemble_primary = _ensemble_config(
        measure_config.primary,
        n_valid=n_valid,
        n_trials=trials.shape[0],
        mode=estimator.mode,
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
        if estimator.mode == "pooled":
            if samples.shape[0] != n_valid:
                raise RuntimeError(f"measure {name} does not preserve the VAR batch axis")
            estimate = estimate[0] if estimate.shape[:1] == (1,) else estimate
            sample_view = samples
        else:
            expected = n_valid * trials.shape[0]
            if samples.shape[0] != expected:
                raise RuntimeError(f"measure {name} does not preserve the VAR batch axis")
            sample_view = samples.reshape(n_valid, trials.shape[0], *samples.shape[1:])
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
        var_order=estimator.order,
        fit_mode=estimator.mode,
    )


__all__ = ["InferenceMeasureConfig", "measure_confidence_intervals"]
