"""Primary analytical measures derived from VAR and state-space parameters.

Generating and fitted models are evaluated through the same code path. Shared
representations such as stationary covariance, autocovariances, spectra and
innovations state-space forms are built once and reused across measure families.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import torch

from ..control import (
    InnovationsStateSpace,
    dynamical_dependence,
    innovations_form,
    stochastic_interaction,
    var_to_innovations_state_space,
)
from ..representations import LinearDynamicalSystem, VARSystem
from .cmem import compute_cmem
from .criticality import covariance_amplification, dominant_timescale, stability_margin
from .dynamics import (
    active_information_storage,
    cross_spectral_density,
    entropy_rate,
    predictive_information,
    spectral_entropy,
)
from .emergence import emergence_measures
from .gaussian import (
    dual_total_correlation,
    gaussian_entropy,
    gaussian_mutual_information,
    o_information,
    s_information,
    total_correlation,
)
from .mvgc import (
    state_space_spectral_mvgc,
    state_space_temporal_mvgc,
)
from .phid import gaussian_phiid_atoms

Model = VARSystem | LinearDynamicalSystem | InnovationsStateSpace
CovarianceModel = VARSystem | LinearDynamicalSystem


@dataclass(frozen=True)
class ModelMeasureConfig:
    """Structural choices for analytical measures.

    Delay-bearing families have independent parameters. Their defaults are one
    sample, while measures defined from the complete model history (MVGC,
    entropy rate, predictive information and AIS) do not receive an artificial
    delay parameter.
    """

    frequencies: torch.Tensor | None = None
    autocovariance_max_lag: int = 1
    cmem_max_lag: int = 1
    source: tuple[int, ...] | None = None
    target: tuple[int, ...] | None = None
    conditional: tuple[int, ...] = ()
    phiid_variables: tuple[int, int] | None = None
    phiid_lag: int = 1
    macro_projection: torch.Tensor | None = None
    partition: tuple[tuple[int, ...], ...] | None = None
    base: float = 2.0

    def __post_init__(self) -> None:
        for name in ("autocovariance_max_lag", "cmem_max_lag", "phiid_lag"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least one")


@dataclass(frozen=True)
class ModelMeasureContext:
    """Shared analytical representations computed once for one model."""

    model: Model
    observation_covariance: torch.Tensor | None
    autocovariances: torch.Tensor | None
    innovations: InnovationsStateSpace | None
    frequencies: torch.Tensor | None
    cross_spectral_density: torch.Tensor | None

    @property
    def max_lag(self) -> int:
        return 0 if self.autocovariances is None else self.autocovariances.shape[1] - 1


def _batched(tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
    value = torch.as_tensor(tensor)
    single = value.ndim == 2
    return (value.unsqueeze(0) if single else value), single


def model_autocovariances(model: CovarianceModel, max_lag: int = 1) -> torch.Tensor:
    """Return Gamma[tau]=Cov(y[t+tau], y[t]) for all lags in one pass."""
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
        transition, single = _batched(model.transition)
        observation, _ = _batched(model.observation)
        state_covariance, _ = _batched(model.state_covariance)
        observation_noise, _ = _batched(model.observation_covariance)
        batch = max(
            transition.shape[0],
            observation.shape[0],
            state_covariance.shape[0],
            observation_noise.shape[0],
        )
        transition, observation, state_covariance, observation_noise = [
            value.expand(batch, *value.shape[1:]) if value.shape[0] == 1 else value
            for value in (transition, observation, state_covariance, observation_noise)
        ]

    batch = transition.shape[0]
    state_identity = torch.eye(
        transition.shape[-1], dtype=transition.dtype, device=transition.device
    ).expand(batch, -1, -1)
    power = state_identity
    values = []
    for lag in range(max_lag + 1):
        if lag:
            power = power @ transition
        gamma = observation @ power @ state_covariance @ observation.transpose(-1, -2)
        if lag == 0:
            gamma = gamma + observation_noise
        values.append(gamma)
    result = torch.stack(values, dim=1)
    return result[0] if single else result


def stationary_observation_covariance(model: CovarianceModel) -> torch.Tensor:
    """Return the model-implied stationary covariance of observed variables."""
    return model_autocovariances(model, 0)[:, 0] if isinstance(model, VARSystem) else model_autocovariances(model, 0)[0]


def _as_innovations(model: Model) -> InnovationsStateSpace | None:
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
    return None


def required_autocovariance_max_lag(model: Model, config: ModelMeasureConfig) -> int:
    """Resolve all requested delay-bearing measures to one shared maximum lag."""
    required = [config.autocovariance_max_lag]
    if config.phiid_variables is not None:
        required.append(config.phiid_lag)
    if isinstance(model, VARSystem):
        required.extend([config.cmem_max_lag, model.order])
    return max(required)


def build_measure_context(
    model: Model,
    config: ModelMeasureConfig | None = None,
) -> ModelMeasureContext:
    """Build and cache shared representations for analytical measures."""
    config = ModelMeasureConfig() if config is None else config
    covariance = None
    autocovariance_sequence = None
    spectrum = None
    frequencies = config.frequencies

    if isinstance(model, (VARSystem, LinearDynamicalSystem)):
        max_lag = required_autocovariance_max_lag(model, config)
        autocovariance_sequence = model_autocovariances(model, max_lag)
        covariance = (
            autocovariance_sequence[:, 0]
            if autocovariance_sequence.ndim == 4
            else autocovariance_sequence[0]
        )

    if isinstance(model, VARSystem) and frequencies is not None:
        spectrum = cross_spectral_density(model, frequencies)

    return ModelMeasureContext(
        model=model,
        observation_covariance=covariance,
        autocovariances=autocovariance_sequence,
        innovations=_as_innovations(model),
        frequencies=frequencies,
        cross_spectral_density=spectrum,
    )


def pairwise_gaussian_mutual_information(
    covariance: torch.Tensor, *, base: float = 2.0
) -> torch.Tensor:
    """Pairwise Gaussian MI matrix from a model-implied covariance."""
    covariance = torch.as_tensor(covariance)
    n_variables = covariance.shape[-1]
    result = torch.zeros(
        (*covariance.shape[:-2], n_variables, n_variables),
        dtype=covariance.dtype,
        device=covariance.device,
    )
    for left in range(n_variables):
        for right in range(left + 1, n_variables):
            index = torch.tensor([left, right], dtype=torch.long, device=covariance.device)
            value = gaussian_mutual_information(
                covariance.index_select(-2, index).index_select(-1, index),
                1,
                base=base,
            )
            result[..., left, right] = value
            result[..., right, left] = value
    return result


def gaussian_measures_from_model(
    model: CovarianceModel,
    *,
    base: float = 2.0,
    covariance: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Static Gaussian measures from the model-implied observation covariance."""
    covariance = stationary_observation_covariance(model) if covariance is None else covariance
    return {
        "covariance": covariance,
        "entropy": gaussian_entropy(covariance, base=base),
        "pairwise_mutual_information": pairwise_gaussian_mutual_information(covariance, base=base),
        "total_correlation": total_correlation(covariance, base=base),
        "dual_total_correlation": dual_total_correlation(covariance, base=base),
        "o_information": o_information(covariance, base=base),
        "s_information": s_information(covariance, base=base),
    }


def past_future_covariance(
    model: CovarianceModel,
    variables: tuple[int, int],
    *,
    lag: int = 1,
    autocovariance_sequence: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build [past0,past1,future0,future1] covariance analytically."""
    if lag < 1:
        raise ValueError("lag must be at least one")
    if len(variables) != 2 or variables[0] == variables[1]:
        raise ValueError("variables must contain two distinct indices")
    gamma = (
        model_autocovariances(model, lag)
        if autocovariance_sequence is None
        else autocovariance_sequence
    )
    if gamma.shape[-3] <= lag:
        raise ValueError("autocovariance_sequence does not contain the requested lag")
    if gamma.ndim == 3:
        gamma = gamma.unsqueeze(0)
        single = True
    else:
        single = False
    index = torch.as_tensor(variables, dtype=torch.long, device=gamma.device)
    present = gamma[:, 0].index_select(-2, index).index_select(-1, index)
    future_past = gamma[:, lag].index_select(-2, index).index_select(-1, index)
    top = torch.cat([present, future_past.transpose(-1, -2)], dim=-1)
    bottom = torch.cat([future_past, present], dim=-1)
    result = torch.cat([top, bottom], dim=-2)
    return result[0] if single else result


def phiid_from_model(
    model: CovarianceModel,
    variables: tuple[int, int],
    *,
    lag: int = 1,
    autocovariance_sequence: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Complete Gaussian MMI PhiID from model-implied lagged covariance."""
    return gaussian_phiid_atoms(
        past_future_covariance(
            model,
            variables,
            lag=lag,
            autocovariance_sequence=autocovariance_sequence,
        )
    )


def temporal_mvgc(
    model: VARSystem | InnovationsStateSpace | LinearDynamicalSystem,
    source,
    target,
    *,
    conditional=(),
    base: float = torch.e,
) -> torch.Tensor:
    """Canonical primary temporal MVGC derived from model parameters."""
    innovations = _as_innovations(model)
    if innovations is None:
        raise TypeError("unsupported model type")
    return state_space_temporal_mvgc(
        innovations,
        source=source,
        target=target,
        conditional=conditional,
        base=float(base),
    )


def spectral_mvgc(
    model: VARSystem | InnovationsStateSpace | LinearDynamicalSystem,
    source,
    target,
    frequencies: torch.Tensor,
    *,
    conditional=(),
    base: float = torch.e,
) -> torch.Tensor:
    """Canonical primary spectral MVGC derived from model parameters."""
    innovations = _as_innovations(model)
    if innovations is None:
        raise TypeError("unsupported model type")
    return state_space_spectral_mvgc(
        innovations,
        source=source,
        target=target,
        conditional=conditional,
        frequencies=frequencies,
        base=float(base),
    )


def compute_all_model_measures(
    model: Model,
    config: ModelMeasureConfig | None = None,
    *,
    context: ModelMeasureContext | None = None,
) -> dict[str, Any]:
    """Compute every primary measure applicable to a canonical model."""
    config = ModelMeasureConfig() if config is None else config
    context = build_measure_context(model, config) if context is None else context
    if context.model is not model:
        raise ValueError("context was built for a different model")

    result: dict[str, Any] = {
        "model_type": type(model).__name__,
        "context": context,
    }

    if isinstance(model, (VARSystem, LinearDynamicalSystem)):
        result["gaussian"] = gaussian_measures_from_model(
            model,
            base=config.base,
            covariance=context.observation_covariance,
        )
        result["autocovariances"] = context.autocovariances

        if config.phiid_variables is not None:
            result["phiid"] = phiid_from_model(
                model,
                config.phiid_variables,
                lag=config.phiid_lag,
                autocovariance_sequence=context.autocovariances,
            )

    if isinstance(model, VARSystem):
        result["dynamics"] = {
            "entropy_rate": entropy_rate(model, base=config.base),
            "predictive_information": predictive_information(model, base=config.base),
            "active_information_storage": active_information_storage(model, base=config.base),
        }
        result["criticality"] = {
            "spectral_radius": model.spectral_radius,
            "stability_margin": stability_margin(model),
            "dominant_timescale": dominant_timescale(model),
            "covariance_amplification": covariance_amplification(model),
        }
        result["cmem"] = compute_cmem(
            model,
            config.cmem_max_lag,
            autocovariance_sequence=context.autocovariances,
        )
        if config.frequencies is not None:
            result["frequency"] = {
                "cross_spectral_density": context.cross_spectral_density,
                "spectral_entropy": spectral_entropy(model, config.frequencies),
            }
        if config.macro_projection is not None:
            result["emergence"] = emergence_measures(model, config.macro_projection)

    if config.source is not None and config.target is not None:
        result["mvgc"] = {
            "temporal": temporal_mvgc(
                model,
                source=config.source,
                target=config.target,
                conditional=config.conditional,
                base=config.base,
            )
        }
        if config.frequencies is not None:
            result["mvgc"]["spectral"] = spectral_mvgc(
                model,
                source=config.source,
                target=config.target,
                conditional=config.conditional,
                frequencies=config.frequencies,
                base=config.base,
            )

    if isinstance(model, LinearDynamicalSystem):
        result["control"] = {
            "dynamical_dependence": dynamical_dependence(model, base=config.base),
        }
        if config.partition is not None:
            result["control"]["stochastic_interaction"] = stochastic_interaction(
                model,
                config.partition,
                base=config.base,
            )

    return result


__all__ = [
    "Model",
    "CovarianceModel",
    "ModelMeasureConfig",
    "ModelMeasureContext",
    "model_autocovariances",
    "required_autocovariance_max_lag",
    "build_measure_context",
    "stationary_observation_covariance",
    "pairwise_gaussian_mutual_information",
    "gaussian_measures_from_model",
    "past_future_covariance",
    "phiid_from_model",
    "temporal_mvgc",
    "spectral_mvgc",
    "compute_all_model_measures",
]
