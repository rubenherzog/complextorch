"""Canonical analytical primitives shared by VAR and state-space measures.

Notes
-----
The canonical backbone maps VAR and linear state-space models to shared
covariance, innovation, spectral and autocovariance primitives. Measures are
then computed from these invariants rather than duplicated by model class.

References
----------
- Barnett, L. and Seth, A. K. (2015), state-space Granger causality.
- Lütkepohl, H. (2005), companion-form VAR representations.
"""
from __future__ import annotations

import math
import torch

from ..control import (
    InnovationsStateSpace,
    innovations_form,
    innovations_transfer_function,
    solve_generalized_dare,
    var_to_innovations_state_space,
)
from ..linalg import spd_logdet, spd_solve, symmetrise
from ..representations import LinearDynamicalSystem, VARSystem
from .gaussian import gaussian_entropy, gaussian_mutual_information, total_correlation

Model = VARSystem | LinearDynamicalSystem | InnovationsStateSpace
CovarianceModel = VARSystem | LinearDynamicalSystem


def as_innovations(model: Model) -> InnovationsStateSpace:
    """Return the exact steady-state innovations representation of a model."""
    if isinstance(model, InnovationsStateSpace):
        return model
    if isinstance(model, VARSystem):
        return var_to_innovations_state_space(model)
    if isinstance(model, LinearDynamicalSystem):
        form = innovations_form(model)
        return InnovationsStateSpace(
            model.transition,
            model.observation,
            form.gain,
            form.covariance,
        )
    raise TypeError("unsupported model type")


def _batched_matrix(value: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """ batched matrix.
    
    Parameters
    ----------
    value
        Input controlling ``_batched_matrix``.
    
    Returns
    -------
    object
        Result described by the function name and annotated return type.
    
    Notes
    -----
    Tensor batch dimensions are preserved unless the public API explicitly
    documents a squeeze operation. Numerical validation is performed by the
    module before the core calculation.
    """
    value = torch.as_tensor(value)
    single = value.ndim == 2
    if single:
        value = value.unsqueeze(0)
    if value.ndim != 3:
        raise ValueError("matrix must be unbatched or batched")
    return value, single


def observation_autocovariances(model: CovarianceModel, max_lag: int = 1) -> torch.Tensor:
    """Return Gamma[tau]=Cov(y[t+tau],y[t]) for lags 0..max_lag."""
    if max_lag < 0:
        raise ValueError("max_lag must be nonnegative")
    if isinstance(model, VARSystem):
        transition = model.companion
        observation = model.projection
        state_covariance = model.state_covariance
        observation_noise = torch.zeros_like(model.present_covariance)
        single = False
    else:
        if model.state_covariance is None:
            raise ValueError("LinearDynamicalSystem.state_covariance is required")
        transition, transition_single = _batched_matrix(model.transition)
        observation, observation_single = _batched_matrix(model.observation)
        state_covariance, covariance_single = _batched_matrix(model.state_covariance)
        observation_noise, noise_single = _batched_matrix(model.observation_covariance)
        single = transition_single and observation_single and covariance_single and noise_single
        batch = max(
            transition.shape[0], observation.shape[0],
            state_covariance.shape[0], observation_noise.shape[0]
        )
        transition, observation, state_covariance, observation_noise = [
            value.expand(batch, *value.shape[1:]) if value.shape[0] == 1 else value
            for value in (transition, observation, state_covariance, observation_noise)
        ]
    batch = transition.shape[0]
    power = torch.eye(
        transition.shape[-1], dtype=transition.dtype, device=transition.device
    ).expand(batch, -1, -1)
    values = []
    for lag in range(max_lag + 1):
        if lag:
            power = power @ transition
        gamma = observation @ power @ state_covariance @ observation.transpose(-1, -2)
        if lag == 0:
            gamma = gamma + observation_noise
        values.append(symmetrise(gamma) if lag == 0 else gamma)
    result = torch.stack(values, dim=1)
    return result[0] if single else result


def innovations_spectrum(
    model: Model,
    frequencies: torch.Tensor,
    *,
    sampling_frequency: float = 1.0,
) -> torch.Tensor:
    """Observation cross-spectrum from the canonical innovations form."""
    innovations = as_innovations(model)
    transfer = innovations_transfer_function(innovations, frequencies)
    covariance = innovations.innovation_covariance
    single = transfer.ndim == 3
    if single:
        transfer = transfer.unsqueeze(0)
        covariance = covariance.unsqueeze(0) if covariance.ndim == 2 else covariance
    spectrum = (
        transfer
        @ covariance[:, None].to(transfer.dtype)
        @ transfer.conj().transpose(-1, -2)
        / float(sampling_frequency)
    )
    return spectrum[0] if single else spectrum


def entropy_rate_from_model(model: Model, *, base: float = 2.0) -> torch.Tensor:
    """Gaussian entropy rate from the innovations covariance."""
    return gaussian_entropy(as_innovations(model).innovation_covariance, base=base)


def predictive_information_from_model(
    model: CovarianceModel,
    *,
    observation_covariance: torch.Tensor | None = None,
    base: float = 2.0,
) -> torch.Tensor:
    """One-step predictive information H(y_t)-H(y_t|past)."""
    covariance = (
        observation_autocovariances(model, 0)[..., 0, :, :]
        if observation_covariance is None else observation_covariance
    )
    innovations = as_innovations(model).innovation_covariance
    # Evaluate log-determinants through an SPD-aware factorisation for numerical stability.
    return 0.5 * (spd_logdet(covariance) - spd_logdet(innovations)) / math.log(base)


def spectral_entropy_from_spectrum(
    spectrum: torch.Tensor,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Per-variable spectral entropy from a cross-spectral density."""
    power = torch.diagonal(spectrum, dim1=-2, dim2=-1).real
    power = power.clamp_min(torch.finfo(power.dtype).tiny)
    frequency_axis = -2
    probability = power / power.sum(frequency_axis, keepdim=True)
    value = -(probability * torch.log2(probability)).sum(frequency_axis)
    if normalize:
        value = value / math.log2(probability.shape[frequency_axis])
    return value


def finite_lag_ais(
    autocovariances: torch.Tensor,
    lag: int = 1,
    *,
    base: float = 2.0,
) -> torch.Tensor:
    """Per-variable I(y_t^i; y_{t-1:t-lag}^i) from autocovariances."""
    if lag < 1 or autocovariances.shape[-3] <= lag:
        raise ValueError("autocovariances do not contain the requested AIS lag")
    gamma = autocovariances if autocovariances.ndim == 4 else autocovariances.unsqueeze(0)
    single = autocovariances.ndim == 3
    n = gamma.shape[-1]
    values = []
    for node in range(n):
        history = torch.empty(
            (gamma.shape[0], lag, lag), dtype=gamma.dtype, device=gamma.device
        )
        for left in range(lag):
            for right in range(lag):
                delta = right - left
                if delta >= 0:
                    history[:, left, right] = gamma[:, delta, node, node]
                else:
                    history[:, left, right] = gamma[:, -delta, node, node]
        cross = torch.stack([gamma[:, k, node, node] for k in range(1, lag + 1)], -1)
        current = gamma[:, 0, node, node].reshape(-1, 1, 1)
        joint = torch.cat(
            [
                torch.cat([current, cross.unsqueeze(-2)], -1),
                torch.cat([cross.unsqueeze(-1), history], -1),
            ],
            -2,
        )
        values.append(gaussian_mutual_information(joint, 1, base=base))
    result = torch.stack(values, -1)
    return result[0] if single else result


def covariance_amplification_from_model(
    model: CovarianceModel,
    *,
    observation_covariance: torch.Tensor | None = None,
) -> torch.Tensor:
    """Covariance amplification from model.
    
    Parameters
    ----------
    model
        Input controlling ``covariance_amplification_from_model``.
    observation_covariance
        Input controlling ``covariance_amplification_from_model``.
    
    Returns
    -------
    object
        Result described by the function name and annotated return type.
    
    Notes
    -----
    Tensor batch dimensions are preserved unless the public API explicitly
    documents a squeeze operation. Numerical validation is performed by the
    module before the core calculation.
    """
    covariance = (
        observation_autocovariances(model, 0)[..., 0, :, :]
        if observation_covariance is None else observation_covariance
    )
    innovations = as_innovations(model).innovation_covariance
    return (
        torch.diagonal(covariance, dim1=-2, dim2=-1).sum(-1)
        / torch.diagonal(innovations, dim1=-2, dim2=-1).sum(-1)
    )


def cmem_from_primitives(
    observation_covariance: torch.Tensor,
    innovation_covariance: torch.Tensor,
    autocovariances: torch.Tensor,
    *,
    curve_max_lag: int = 1,
    decomposition_max_lag: int = 1,
):
    """Generic CMem totals, curves and finite-lag decomposition."""
    from .cmem import compute_cmem_from_primitives
    return compute_cmem_from_primitives(
        observation_covariance,
        innovation_covariance,
        autocovariances,
        curve_max_lag=curve_max_lag,
        decomposition_max_lag=decomposition_max_lag,
    )


def projected_innovation_covariance(
    model: Model,
    projection: torch.Tensor,
) -> torch.Tensor:
    """Innovation covariance of a linear projection of the observations."""
    innovations = as_innovations(model)
    a, single = _batched_matrix(innovations.transition)
    c, _ = _batched_matrix(innovations.observation)
    k, _ = _batched_matrix(innovations.gain)
    v, _ = _batched_matrix(innovations.innovation_covariance)
    m = torch.as_tensor(projection, dtype=c.dtype, device=c.device)
    if m.ndim == 2:
        m = m.unsqueeze(0)
    batch = max(a.shape[0], c.shape[0], k.shape[0], v.shape[0], m.shape[0])
    a, c, k, v, m = [
        value.expand(batch, *value.shape[1:]) if value.shape[0] == 1 else value
        for value in (a, c, k, v, m)
    ]
    projected_c = m @ c
    process = k @ v @ k.transpose(-1, -2)
    observation = symmetrise(m @ v @ m.transpose(-1, -2))
    cross = k @ v @ m.transpose(-1, -2)
    # Marginal innovations require the steady-state generalised Riccati solution.
    prediction = solve_generalized_dare(a, projected_c, process, observation, cross)
    if prediction.ndim == 2:
        prediction = prediction.unsqueeze(0)
    result = symmetrise(projected_c @ prediction @ projected_c.transpose(-1, -2) + observation)
    return result[0] if single and projection.ndim == 2 else result


def emergence_from_model(
    model: CovarianceModel,
    macro_projection: torch.Tensor,
    *,
    observation_covariance: torch.Tensor | None = None,
    base: float = 2.0,
) -> dict[str, torch.Tensor]:
    """Gaussian Psi, Delta and Gamma using exact projected innovations."""
    covariance = (
        observation_autocovariances(model, 0)[..., 0, :, :]
        if observation_covariance is None else observation_covariance
    )
    m = torch.as_tensor(macro_projection, dtype=covariance.dtype, device=covariance.device)
    if m.ndim == 2 and covariance.ndim == 3:
        m = m.unsqueeze(0)
    macro_covariance = symmetrise(m @ covariance @ m.transpose(-1, -2))
    micro_innovation = as_innovations(model).innovation_covariance
    macro_given_micro = symmetrise(m @ micro_innovation @ m.transpose(-1, -2))
    macro_innovation = projected_innovation_covariance(model, m)
    # Evaluate log-determinants through an SPD-aware factorisation for numerical stability.
    i_full = 0.5 * (spd_logdet(macro_covariance) - spd_logdet(macro_given_micro)) / math.log(base)
    # Evaluate log-determinants through an SPD-aware factorisation for numerical stability.
    i_macro = 0.5 * (spd_logdet(macro_covariance) - spd_logdet(macro_innovation)) / math.log(base)
    full_parts = 0.5 * torch.log(
        torch.diagonal(macro_covariance, dim1=-2, dim2=-1)
        / torch.diagonal(macro_given_micro, dim1=-2, dim2=-1)
    ).sum(-1) / math.log(base)
    self_parts = 0.5 * torch.log(
        torch.diagonal(macro_covariance, dim1=-2, dim2=-1)
        / torch.diagonal(macro_innovation, dim1=-2, dim2=-1)
    ).sum(-1) / math.log(base)
    return {
        "psi": i_full - i_macro,
        "delta": i_macro - self_parts,
        "gamma": i_full - full_parts,
        "macro_predictive_information": i_macro,
        "macro_from_micro_predictive_information": i_full,
    }


__all__ = [
    "Model", "CovarianceModel", "as_innovations", "observation_autocovariances",
    "innovations_spectrum", "entropy_rate_from_model",
    "predictive_information_from_model", "spectral_entropy_from_spectrum",
    "finite_lag_ais", "covariance_amplification_from_model",
    "cmem_from_primitives", "projected_innovation_covariance",
    "emergence_from_model",
]
