"""Public in-sample and out-of-sample fit-diagnostic evaluation modes."""
from __future__ import annotations

import torch

from .control import _as_innovations_state_space
from .diagnostics import (
    FitDiagnostics,
    _expand_matrix,
    _innovation_covariance,
    _normalise_observations,
    fit_diagnostics as _fit_diagnostics_oos,
    innovation_diagnostics,
)
from .state_space import LarimoreStateSpace, N4SID
from .var import VAR


def _ss_in_sample_errors(estimator, data: torch.Tensor):
    """Return aligned in-sample innovations from the fitted innovations recursion."""
    system = _as_innovations_state_space(estimator.system_)
    batch = data.shape[0]
    transition = _expand_matrix(system.transition, batch).to(data)
    observation = _expand_matrix(system.observation, batch).to(data)
    gain = _expand_matrix(system.gain, batch).to(data)
    training_mean = data.mean(dim=1, keepdim=True)
    centered = data - training_mean
    warmup = int(
        estimator.past_horizon
        if isinstance(estimator, LarimoreStateSpace)
        else estimator.block_rows
    )
    if warmup >= data.shape[1]:
        raise ValueError(
            "state-space identification horizon leaves no diagnostic samples"
        )
    state = torch.zeros(
        batch, transition.shape[-1], dtype=data.dtype, device=data.device
    )
    errors = []
    aligned = []
    for time_index in range(centered.shape[1]):
        innovation = centered[:, time_index] - torch.einsum(
            "bmd,bd->bm", observation, state
        )
        if time_index >= warmup:
            errors.append(innovation)
            aligned.append(data[:, time_index])
        state = torch.einsum("bij,bj->bi", transition, state) + torch.einsum(
            "bdm,bm->bd", gain, innovation
        )
    return torch.stack(aligned, 1), torch.stack(errors, 1), training_mean


def fit_diagnostics(
    estimator,
    train,
    test=None,
    *,
    evaluation: str = "oos",
    max_lag: int = 10,
) -> FitDiagnostics:
    """Evaluate fitted VAR/state-space models in- or out-of-sample.

    ``evaluation="oos"`` preserves the original rolling held-out contract.
    ``evaluation="in_sample"`` computes one-step residual diagnostics on the
    supplied fitted-data trajectory without refitting the estimator. Both modes
    delegate all statistics to :func:`innovation_diagnostics`, so
    ``whiteness_energy`` has exactly the same definition in both regimes.
    """
    if evaluation not in {"oos", "in_sample"}:
        raise ValueError("evaluation must be 'oos' or 'in_sample'")
    if evaluation == "oos":
        if test is None:
            raise ValueError("test is required for evaluation='oos'")
        return _fit_diagnostics_oos(estimator, train, test, max_lag=max_lag)
    if test is not None:
        raise ValueError("test must be omitted for evaluation='in_sample'")

    data, single = _normalise_observations(train)
    mode = getattr(estimator, "mode", "pooled")
    training_mean = data.mean(dim=1, keepdim=True)
    if isinstance(estimator, VAR):
        if not hasattr(estimator, "coef_"):
            raise ValueError("estimator must be fitted before diagnostics")
        if data.shape[1] <= estimator.order:
            raise ValueError("observations are too short for the fitted VAR order")
        errors = estimator.residuals(data).to(data)
        aligned = data[:, estimator.order :]
    elif isinstance(estimator, (LarimoreStateSpace, N4SID)):
        if not hasattr(estimator, "system_"):
            raise ValueError("estimator must be fitted before diagnostics")
        aligned, errors, training_mean = _ss_in_sample_errors(estimator, data)
    else:
        raise TypeError(
            "estimator must be a fitted VAR, LarimoreStateSpace, or N4SID"
        )

    covariance = _innovation_covariance(estimator, aligned.shape[0], aligned)
    return innovation_diagnostics(
        aligned[0] if single else aligned,
        errors[0] if single else errors,
        covariance[0] if single else covariance,
        training_mean=training_mean[0, 0] if single else training_mean[:, 0],
        max_lag=max_lag,
        mode=mode,
    )


__all__ = ["fit_diagnostics"]
