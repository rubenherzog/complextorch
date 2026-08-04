"""Primary analytical API derived from VAR and state-space model parameters.

The functions in this module never estimate a model from observations.  They
consume canonical model objects and derive all intermediate covariances,
autocovariances and spectra from those parameters so ground-truth and fitted
models can be evaluated through exactly the same code path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import torch

from ..control import InnovationsStateSpace, dynamical_dependence, stochastic_interaction
from ..representations import LinearDynamicalSystem, VARSystem
from .cmem import compute_cmem
from .criticality import covariance_amplification, dominant_timescale, stability_margin
from .dynamics import (
    active_information_storage,
    autocovariances,
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
from .mvgc import state_space_spectral_mvgc, state_space_temporal_mvgc
from .phid import gaussian_phiid_atoms


Model = VARSystem | LinearDynamicalSystem | InnovationsStateSpace


@dataclass(frozen=True)
class ModelMeasureConfig:
    """Optional structure required by non-scalar model measures."""

    frequencies: torch.Tensor | None = None
    max_lag: int = 20
    cmem_tau_max: int = 20
    source: tuple[int, ...] | None = None
    target: tuple[int, ...] | None = None
    conditional: tuple[int, ...] = ()
    phiid_variables: tuple[int, int] | None = None
    phiid_lag: int = 1
    macro_projection: torch.Tensor | None = None
    partition: tuple[tuple[int, ...], ...] | None = None
    base: float = 2.0


def stationary_observation_covariance(model: VARSystem | LinearDynamicalSystem) -> torch.Tensor:
    """Return the stationary covariance of the observed variables."""
    if isinstance(model, VARSystem):
        return model.present_covariance
    if model.state_covariance is None:
        raise ValueError("LinearDynamicalSystem.state_covariance is required")
    return (
        model.observation @ model.state_covariance @ model.observation.transpose(-1, -2)
        + model.observation_covariance
    )


def pairwise_gaussian_mutual_information(
    covariance: torch.Tensor, *, base: float = 2.0
) -> torch.Tensor:
    """Pairwise Gaussian MI matrix derived from a model covariance."""
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
    model: VARSystem | LinearDynamicalSystem, *, base: float = 2.0
) -> dict[str, torch.Tensor]:
    """Static Gaussian measures from the model-implied observation covariance."""
    covariance = stationary_observation_covariance(model)
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
    system: VARSystem,
    variables: tuple[int, int],
    *,
    lag: int = 1,
) -> torch.Tensor:
    """Build [past0,past1,future0,future1] covariance analytically."""
    if lag < 1:
        raise ValueError("lag must be at least one")
    if len(variables) != 2 or variables[0] == variables[1]:
        raise ValueError("variables must contain two distinct indices")
    index = torch.as_tensor(variables, dtype=torch.long, device=system.coefficients.device)
    gamma = autocovariances(system, lag)
    present = gamma[:, 0].index_select(-2, index).index_select(-1, index)
    future_past = gamma[:, lag].index_select(-2, index).index_select(-1, index)
    top = torch.cat([present, future_past.transpose(-1, -2)], dim=-1)
    bottom = torch.cat([future_past, present], dim=-1)
    return torch.cat([top, bottom], dim=-2)


def phiid_from_model(
    system: VARSystem,
    variables: tuple[int, int],
    *,
    lag: int = 1,
) -> dict[str, torch.Tensor]:
    """Complete Gaussian MMI PhiID from model-implied lagged covariance."""
    return gaussian_phiid_atoms(past_future_covariance(system, variables, lag=lag))


def compute_all_model_measures(
    model: Model,
    config: ModelMeasureConfig | None = None,
) -> dict[str, Any]:
    """Compute every primary measure applicable to a canonical model.

    Optional families are included only when their required structural choices
    are supplied in ``config``.  No quantity is estimated directly from samples.
    """
    config = ModelMeasureConfig() if config is None else config
    result: dict[str, Any] = {"model_type": type(model).__name__}

    if isinstance(model, (VARSystem, LinearDynamicalSystem)):
        result["gaussian"] = gaussian_measures_from_model(model, base=config.base)

    if isinstance(model, VARSystem):
        result["dynamics"] = {
            "autocovariances": autocovariances(model, config.max_lag),
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
        result["cmem"] = compute_cmem(model, config.cmem_tau_max)

        if config.frequencies is not None:
            result["frequency"] = {
                "cross_spectral_density": cross_spectral_density(model, config.frequencies),
                "spectral_entropy": spectral_entropy(model, config.frequencies),
            }

        if config.source is not None and config.target is not None:
            mvgc = {
                "temporal": state_space_temporal_mvgc(
                    model,
                    source=config.source,
                    target=config.target,
                    conditional=config.conditional,
                    base=config.base,
                )
            }
            if config.frequencies is not None:
                mvgc["spectral"] = state_space_spectral_mvgc(
                    model,
                    source=config.source,
                    target=config.target,
                    conditional=config.conditional,
                    frequencies=config.frequencies,
                    base=config.base,
                )
            result["mvgc"] = mvgc

        if config.phiid_variables is not None:
            result["phiid"] = phiid_from_model(
                model,
                config.phiid_variables,
                lag=config.phiid_lag,
            )

        if config.macro_projection is not None:
            result["emergence"] = emergence_measures(model, config.macro_projection)

        state_space = model.to_state_space()
        result["control"] = {
            "dynamical_dependence": dynamical_dependence(state_space, base=config.base),
        }
        if config.partition is not None:
            result["control"]["stochastic_interaction"] = stochastic_interaction(
                state_space,
                config.partition,
                base=config.base,
            )

    elif isinstance(model, LinearDynamicalSystem):
        result["control"] = {
            "dynamical_dependence": dynamical_dependence(model, base=config.base),
        }
        if config.partition is not None:
            result["control"]["stochastic_interaction"] = stochastic_interaction(
                model,
                config.partition,
                base=config.base,
            )

    elif isinstance(model, InnovationsStateSpace):
        if config.source is None or config.target is None:
            raise ValueError(
                "source and target are required to evaluate an InnovationsStateSpace"
            )
        mvgc = {
            "temporal": state_space_temporal_mvgc(
                model,
                source=config.source,
                target=config.target,
                conditional=config.conditional,
                base=config.base,
            )
        }
        if config.frequencies is not None:
            mvgc["spectral"] = state_space_spectral_mvgc(
                model,
                source=config.source,
                target=config.target,
                conditional=config.conditional,
                frequencies=config.frequencies,
                base=config.base,
            )
        result["mvgc"] = mvgc

    return result


__all__ = [
    "Model",
    "ModelMeasureConfig",
    "stationary_observation_covariance",
    "pairwise_gaussian_mutual_information",
    "gaussian_measures_from_model",
    "past_future_covariance",
    "phiid_from_model",
    "compute_all_model_measures",
]
