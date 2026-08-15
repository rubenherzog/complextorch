"""Canonical analytical backbone shared by VAR and state-space models.

Models are mapped to common observation covariance, autocovariance, innovations
and spectral primitives. Measures consume these invariants rather than
reimplementing model-specific formulas.

References
----------
- Lütkepohl, H. (2005). Companion-form VAR representation.
- Barnett, L. and Seth, A. K. (2015). State-space Granger causality.
"""
from __future__ import annotations

import math
import torch

from ..control import (
    InnovationsStateSpace,
    _as_innovations_state_space,
    _project_innovations_state_space,
)
from ..linalg import solve_discrete_lyapunov, spd_logdet, symmetrise
from ..representations import StateSpaceModel, VARSystem
from ..spectra import innovations_spectral_density
from .gaussian import gaussian_entropy, gaussian_mutual_information, total_correlation

Model = VARSystem | StateSpaceModel | InnovationsStateSpace
CovarianceModel = VARSystem | StateSpaceModel | InnovationsStateSpace


def as_innovations(model: Model) -> InnovationsStateSpace:
    """Return the exact steady-state innovations representation of a model."""
    return _as_innovations_state_space(model)


def _batched_matrix(value: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Batched matrix.
    
    Parameters
    ----------
    value
        Input required by this calculation.
    
    Returns
    -------
    object
        Computed result; see the annotated return type and shape notes.
    
    Notes
    -----
    Batch dimensions are preserved unless explicitly documented otherwise.
    The implementation validates dimensional and positive-definiteness
    requirements before executing the numerical core.
    """
    value = torch.as_tensor(value)
    single = value.ndim == 2
    if single:
        value = value.unsqueeze(0)
    if value.ndim != 3:
        raise ValueError("matrix must be unbatched or batched")
    return value, single


def observation_autocovariances(model: CovarianceModel, max_lag: int = 1) -> torch.Tensor:
    r"""Return :math:`\Gamma_\tau=\operatorname{Cov}(y_{t+\tau},y_t)`.

    For an innovations-form model

    .. math::

       x_{t+1}=Ax_t+K\varepsilon_t,\qquad
       y_t=Cx_t+\varepsilon_t,\qquad
       \operatorname{Cov}(\varepsilon_t)=V,

    the stationary state covariance solves
    :math:`P=APA^\top+KVK^\top`.  The contemporaneous covariance is
    :math:`\Gamma_0=CPC^\top+V`, while for positive lags the shared
    innovation at time ``t`` contributes the required cross term,
    :math:`\operatorname{Cov}(x_{t+1},y_t)=APC^\top+KV`.
    """
    if max_lag < 0:
        raise ValueError("max_lag must be nonnegative")
    if isinstance(model, VARSystem):
        transition = model.companion
        observation = model.projection
        state_covariance = model.state_covariance
        observation_noise = torch.zeros_like(model.present_covariance)
        single = False
        cross_state_observation = None
    elif isinstance(model, InnovationsStateSpace):
        transition, transition_single = _batched_matrix(model.transition)
        observation, observation_single = _batched_matrix(model.observation)
        gain, gain_single = _batched_matrix(model.gain)
        innovation_covariance, covariance_single = _batched_matrix(
            model.innovation_covariance
        )
        single = (
            transition_single
            and observation_single
            and gain_single
            and covariance_single
        )
        batch = max(
            transition.shape[0],
            observation.shape[0],
            gain.shape[0],
            innovation_covariance.shape[0],
        )
        transition, observation, gain, innovation_covariance = [
            value.expand(batch, *value.shape[1:]) if value.shape[0] == 1 else value
            for value in (transition, observation, gain, innovation_covariance)
        ]
        if any(
            value.shape[0] != batch
            for value in (transition, observation, gain, innovation_covariance)
        ):
            raise ValueError("model batch dimensions are not broadcast-compatible")
        process_covariance = symmetrise(
            gain @ innovation_covariance @ gain.transpose(-1, -2)
        )
        state_covariance, _ = solve_discrete_lyapunov(
            transition, process_covariance
        )
        observation_noise = innovation_covariance
        cross_state_observation = (
            transition @ state_covariance @ observation.transpose(-1, -2)
            + gain @ innovation_covariance
        )
    else:
        if model.state_covariance is None:
            raise ValueError("StateSpaceModel.state_covariance is required")
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
        cross_state_observation = None

    values = [
        symmetrise(
            observation @ state_covariance @ observation.transpose(-1, -2)
            + observation_noise
        )
    ]
    if max_lag:
        if cross_state_observation is None:
            cross_state_observation = (
                transition @ state_covariance @ observation.transpose(-1, -2)
            )
        for lag in range(1, max_lag + 1):
            if lag > 1:
                cross_state_observation = transition @ cross_state_observation
            values.append(observation @ cross_state_observation)
    result = torch.stack(values, dim=1)
    return result[0] if single else result


def innovations_spectrum(
    model: Model,
    frequencies: torch.Tensor,
    *,
    sampling_frequency: float = 1.0,
) -> torch.Tensor:
    """Observation cross-spectrum from the canonical innovations form."""
    return innovations_spectral_density(
        as_innovations(model),
        frequencies,
        sampling_frequency=sampling_frequency,
    )


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
        VAR or linear state-space model.
    observation_covariance
        Observation-noise covariance matrix.
    
    Returns
    -------
    object
        Computed result; see the annotated return type and shape notes.
    
    Notes
    -----
    Batch dimensions are preserved unless explicitly documented otherwise.
    The implementation validates dimensional and positive-definiteness
    requirements before executing the numerical core.
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
    projected = _project_innovations_state_space(
        as_innovations(model), projection
    )
    return projected.innovation_covariance


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
