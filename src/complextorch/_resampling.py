r"""Private Torch-native VAR resampling engine for confidence intervals.

The engine implements the ARres-style resampling layer only: fit-derived
innovation resampling, recursive synthetic trajectory generation, batched OLS
refitting, stationary-replicate filtering, and construction of one canonical
batched :class:`~complextorch.VARSystem`. It contains no measure-specific logic.

References
----------
- Beda, A., Simpson, D. M., and Faes, L. (2017). Estimation of confidence
  limits for descriptive indexes derived from autoregressive analysis of time
  series. *PLoS ONE*, 12(10), e0186694.
- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import torch

from .measures.primary import ModelMeasureConfig
from .representations import VARSystem, build_var_system, companion_matrix
from .var import VAR

ResamplingMethod = Literal["residual_bootstrap", "parametric"]


@dataclass(frozen=True)
class MeasureInterval:
    """Point estimate and percentile confidence limits for one measure."""

    estimate: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    samples: torch.Tensor | None
    n_valid: int


@dataclass(frozen=True)
class ConfidenceIntervalResult:
    """Confidence intervals obtained from one shared model resampling ensemble."""

    intervals: Mapping[str, MeasureInterval]
    method: ResamplingMethod
    confidence: float
    n_resamples: int
    n_valid: int
    n_failed: int
    seed: int | None
    var_order: int | None
    fit_mode: str
    estimator_type: str = "VAR"

    def __getitem__(self, name: str) -> MeasureInterval:
        """Return one named interval."""
        return self.intervals[name]


def _normalise_trials(estimator: VAR, x: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Normalize observations through the VAR estimator input contract."""
    return estimator._normalise_input(x)  # noqa: SLF001 - shared estimator contract


def _prediction_residuals(estimator: VAR, trials: torch.Tensor) -> torch.Tensor:
    """Return within-trajectory one-step residual vectors."""
    return trials[:, estimator.order :, :] - estimator.one_step_predictions(trials)


def _draw_innovations(
    residuals: torch.Tensor,
    covariance: torch.Tensor,
    *,
    mode: str,
    method: ResamplingMethod,
    n_resamples: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Draw innovation vectors while preserving zero-lag multivariate structure."""
    batch, length, n_variables = residuals.shape
    if method == "residual_bootstrap":
        if mode == "pooled":
            pool = residuals.reshape(batch * length, n_variables)
            pool = pool - pool.mean(dim=0, keepdim=True)
            index = torch.randint(
                pool.shape[0],
                (n_resamples, batch, length),
                device=residuals.device,
                generator=generator,
            )
            return pool[index]
        centered = residuals - residuals.mean(dim=1, keepdim=True)
        index = torch.randint(
            length,
            (n_resamples, batch, length),
            device=residuals.device,
            generator=generator,
        )
        expanded = centered.unsqueeze(0).expand(n_resamples, -1, -1, -1)
        return torch.gather(
            expanded,
            2,
            index.unsqueeze(-1).expand(-1, -1, -1, n_variables),
        )
    if method != "parametric":
        raise ValueError("method must be 'residual_bootstrap' or 'parametric'")
    covariance = covariance if covariance.ndim == 3 else covariance.unsqueeze(0)
    if mode == "pooled":
        covariance = covariance.expand(batch, -1, -1)
    elif covariance.shape[0] != batch:
        raise ValueError("independent VAR covariance batch does not match trajectories")
    chol = torch.linalg.cholesky(covariance)
    standard = torch.randn(
        (n_resamples, batch, length, n_variables),
        dtype=residuals.dtype,
        device=residuals.device,
        generator=generator,
    )
    return torch.einsum("rbtn,bmn->rbtm", standard, chol)


def _simulate_resamples(
    trials: torch.Tensor,
    estimator: VAR,
    innovations: torch.Tensor,
) -> torch.Tensor:
    """Simulate all resamples in parallel over resample and trajectory axes."""
    n_resamples, batch, length, n_variables = innovations.shape
    order = estimator.order
    output = torch.empty(
        (n_resamples, batch, trials.shape[1], n_variables),
        dtype=trials.dtype,
        device=trials.device,
    )
    output[:, :, :order, :] = trials[None, :, :order, :]
    coefficients = estimator.coef_
    intercept = estimator.intercept_
    if estimator.mode == "pooled":
        coefficients = coefficients.expand(batch, -1, -1, -1)
        intercept = intercept.expand(batch, -1)
    elif coefficients.shape[0] != batch:
        raise ValueError("independent fitted VAR batch does not match trajectories")
    # Time is sequential by definition; resample and trajectory axes stay batched.
    for offset in range(length):
        time = order + offset
        previous = torch.stack(
            [output[:, :, time - lag, :] for lag in range(1, order + 1)],
            dim=-2,
        )
        prediction = torch.einsum("rbps,bpns->rbn", previous, coefficients)
        output[:, :, time, :] = (
            prediction + intercept.unsqueeze(0) + innovations[:, :, offset, :]
        )
    return output


def _batched_ols_refit(
    samples: torch.Tensor,
    estimator: VAR,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Refit all synthetic VARs without creating cross-trajectory lag rows."""
    n_resamples, batch, time, n_variables = samples.shape
    order = estimator.order
    length = time - order
    design = torch.cat(
        [samples[:, :, order - lag : time - lag, :] for lag in range(1, order + 1)],
        dim=-1,
    )
    targets = samples[:, :, order:, :]
    if estimator.fit_intercept:
        design = torch.cat(
            [
                design,
                torch.ones(
                    (*design.shape[:-1], 1),
                    dtype=design.dtype,
                    device=design.device,
                ),
            ],
            dim=-1,
        )
    if estimator.mode == "pooled":
        design_fit = design.reshape(n_resamples, batch * length, -1)
        targets_fit = targets.reshape(n_resamples, batch * length, n_variables)
        output_batch, nfit = n_resamples, batch * length
    elif estimator.mode == "independent":
        design_fit = design.reshape(n_resamples * batch, length, -1)
        targets_fit = targets.reshape(n_resamples * batch, length, n_variables)
        output_batch, nfit = n_resamples * batch, length
    else:
        raise ValueError("mode must be 'pooled' or 'independent'")
    solution = torch.linalg.lstsq(design_fit, targets_fit).solution
    coefficient_flat = solution[:, :-1, :] if estimator.fit_intercept else solution
    coefficients = coefficient_flat.reshape(
        output_batch, order, n_variables, n_variables
    ).transpose(-1, -2)
    residuals = targets_fit - design_fit @ solution
    denominator = nfit if estimator.covariance == "mle" else nfit - design_fit.shape[-1]
    if denominator <= 0:
        raise ValueError("not enough observations for resampled VAR covariance")
    covariance = residuals.transpose(-1, -2) @ residuals / float(denominator)
    return coefficients, 0.5 * (covariance + covariance.transpose(-1, -2))


def _stable_resample_mask(
    coefficients: torch.Tensor,
    *,
    n_resamples: int,
    n_trials: int,
    mode: str,
) -> torch.Tensor:
    """Return a common resample mask containing only stationary fitted VARs."""
    radius = torch.linalg.eigvals(companion_matrix(coefficients)).abs().amax(dim=-1)
    if mode == "pooled":
        return radius < 1
    return (radius.reshape(n_resamples, n_trials) < 1).all(dim=1)


def _build_ensemble(
    coefficients: torch.Tensor,
    covariance: torch.Tensor,
    valid: torch.Tensor,
    *,
    n_resamples: int,
    n_trials: int,
    mode: str,
) -> VARSystem:
    """Build one canonical batched VARSystem from all valid resamples."""
    if mode == "pooled":
        return build_var_system(coefficients[valid], covariance[valid])
    coefficients = coefficients.reshape(
        n_resamples, n_trials, *coefficients.shape[1:]
    )
    covariance = covariance.reshape(n_resamples, n_trials, *covariance.shape[1:])
    return build_var_system(
        coefficients[valid].reshape(-1, *coefficients.shape[2:]),
        covariance[valid].reshape(-1, *covariance.shape[2:]),
    )


def _ensemble_config(
    config: ModelMeasureConfig,
    *,
    n_valid: int,
    n_trials: int,
    mode: str,
) -> ModelMeasureConfig:
    """Broadcast supplied fixed macro projections over the resample axis."""
    projection = config.macro_projection
    if projection is None or projection.ndim == 2:
        return config
    if projection.ndim != 3:
        raise ValueError("macro_projection must have shape (m,n) or (batch,m,n)")
    if mode == "pooled":
        if projection.shape[0] != 1:
            raise ValueError("pooled inference accepts one fixed macro projection")
        return replace(config, macro_projection=projection.expand(n_valid, -1, -1))
    if projection.shape[0] != n_trials:
        raise ValueError("independent inference requires one projection per trajectory")
    expanded = projection.unsqueeze(0).expand(n_valid, -1, -1, -1)
    return replace(
        config,
        macro_projection=expanded.reshape(n_valid * n_trials, *projection.shape[1:]),
    )


__all__ = [
    "ConfidenceIntervalResult",
    "MeasureInterval",
    "ResamplingMethod",
]
