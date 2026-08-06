"""Strict analytical measures computed from supplied generative models.

Primary functions do not refit observations. A shared context computes the
maximum required autocovariance lag and caches reusable model-derived
primitives.

References
----------
- Cover, T. M. and Thomas, J. A. (2006).
- Barnett, L. and Seth, A. K. (2014, 2015).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math
import torch

from ..control import InnovationsStateSpace, dynamical_dependence, stochastic_interaction
from ..representations import StateSpaceModel, VARSystem
from .backbone import (
    CovarianceModel,
    Model,
    as_innovations,
    cmem_from_primitives,
    covariance_amplification_from_model,
    emergence_from_model,
    entropy_rate_from_model,
    finite_lag_ais,
    innovations_spectrum,
    observation_autocovariances,
    predictive_information_from_model,
    spectral_entropy_from_spectrum,
)
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


@dataclass(frozen=True)
class ModelMeasureConfig:
    """Structural choices for model-derived analytical measures."""

    frequencies: torch.Tensor | None = None
    sampling_frequency: float = 1.0
    autocovariance_max_lag: int = 1
    ais_lag: int = 1
    cmem_max_lag: int = 1
    cmem_decomposition_max_lag: int = 1
    source: tuple[int, ...] | None = None
    target: tuple[int, ...] | None = None
    conditional: tuple[int, ...] = ()
    phiid_variables: tuple[int, int] | None = None
    phiid_lag: int = 1
    macro_projection: torch.Tensor | None = None
    partition: tuple[tuple[int, ...], ...] | None = None
    base: float = 2.0

    def __post_init__(self) -> None:
        """Post init.
        
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
        for name in (
            "autocovariance_max_lag",
            "ais_lag",
            "cmem_max_lag",
            "cmem_decomposition_max_lag",
            "phiid_lag",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least one")
        if self.sampling_frequency <= 0:
            raise ValueError("sampling_frequency must be positive")


@dataclass(frozen=True)
class ModelMeasureContext:
    """Canonical representations computed once and shared by all measures."""

    model: Model
    innovations: InnovationsStateSpace
    observation_covariance: torch.Tensor | None
    autocovariances: torch.Tensor | None
    frequencies: torch.Tensor | None
    cross_spectral_density: torch.Tensor | None

    @property
    def max_lag(self) -> int:
        """Max lag.
        
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
        return 0 if self.autocovariances is None else int(self.autocovariances.shape[-3] - 1)


def required_autocovariance_max_lag(model: Model, config: ModelMeasureConfig) -> int:
    """Resolve all finite-delay families to one shared maximum lag."""
    required = [
        config.autocovariance_max_lag,
        config.ais_lag,
        config.cmem_max_lag,
        config.cmem_decomposition_max_lag,
    ]
    if config.phiid_variables is not None:
        required.append(config.phiid_lag)
    return max(required)


def build_measure_context(
    model: Model,
    config: ModelMeasureConfig | None = None,
) -> ModelMeasureContext:
    """Build the canonical analytical backbone for one model."""
    config = ModelMeasureConfig() if config is None else config
    innovations = as_innovations(model)
    covariance = None
    autocovariance_sequence = None
    if isinstance(model, (VARSystem, StateSpaceModel)):
        autocovariance_sequence = observation_autocovariances(
            model, required_autocovariance_max_lag(model, config)
        )
        covariance = autocovariance_sequence[..., 0, :, :]
    spectrum = None
    if config.frequencies is not None:
        spectrum = innovations_spectrum(
            model,
            config.frequencies,
            sampling_frequency=config.sampling_frequency,
        )
    return ModelMeasureContext(
        model=model,
        innovations=innovations,
        observation_covariance=covariance,
        autocovariances=autocovariance_sequence,
        frequencies=config.frequencies,
        cross_spectral_density=spectrum,
    )


def stationary_observation_covariance(model: CovarianceModel) -> torch.Tensor:
    """Stationary observation covariance.
    
    Parameters
    ----------
    model
        VAR or linear state-space model.
    
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
    return observation_autocovariances(model, 0)[..., 0, :, :]


def model_autocovariances(model: CovarianceModel, max_lag: int = 1) -> torch.Tensor:
    """Model autocovariances.
    
    Parameters
    ----------
    model
        VAR or linear state-space model.
    max_lag
        Largest non-negative lag to evaluate.
    
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
    return observation_autocovariances(model, max_lag)


def pairwise_gaussian_mutual_information(
    covariance: torch.Tensor, *, base: float = 2.0
) -> torch.Tensor:
    """Pairwise gaussian mutual information.
    
    Parameters
    ----------
    covariance
        Symmetric covariance matrix or batch of covariance matrices.
    base
        Logarithm base used for information quantities.
    
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
    """Gaussian measures from model.
    
    Parameters
    ----------
    model
        VAR or linear state-space model.
    base
        Logarithm base used for information quantities.
    covariance
        Symmetric covariance matrix or batch of covariance matrices.
    
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
    """Past future covariance.
    
    Parameters
    ----------
    model
        VAR or linear state-space model.
    variables
        Input required by this calculation.
    lag
        Positive temporal lag in samples.
    autocovariance_sequence
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
    if lag < 1:
        raise ValueError("lag must be at least one")
    if len(variables) != 2 or variables[0] == variables[1]:
        raise ValueError("variables must contain two distinct indices")
    gamma = (
        observation_autocovariances(model, lag)
        if autocovariance_sequence is None else autocovariance_sequence
    )
    if gamma.shape[-3] <= lag:
        raise ValueError("autocovariances do not contain the requested lag")
    single = gamma.ndim == 3
    if single:
        gamma = gamma.unsqueeze(0)
    index = torch.as_tensor(variables, dtype=torch.long, device=gamma.device)
    present = gamma[:, 0].index_select(-2, index).index_select(-1, index)
    future_past = gamma[:, lag].index_select(-2, index).index_select(-1, index)
    top = torch.cat([present, future_past.transpose(-1, -2)], -1)
    bottom = torch.cat([future_past, present], -1)
    result = torch.cat([top, bottom], -2)
    return result[0] if single else result


def phiid_from_model(
    model: CovarianceModel,
    variables: tuple[int, int],
    *,
    lag: int = 1,
    autocovariance_sequence: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Phiid from model.
    
    Parameters
    ----------
    model
        VAR or linear state-space model.
    variables
        Input required by this calculation.
    lag
        Positive temporal lag in samples.
    autocovariance_sequence
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
    return gaussian_phiid_atoms(
        past_future_covariance(
            model,
            variables,
            lag=lag,
            autocovariance_sequence=autocovariance_sequence,
        )
    )


def temporal_mvgc(
    model: Model,
    source,
    target,
    *,
    conditional=(),
    base: float = math.e,
) -> torch.Tensor:
    """Compute conditional time-domain multivariate Granger causality.
    
    .. math::
    
       F_{Y\to X\mid Z}
       =\log\frac{\det\Sigma^{R}_{XX}}{\det\Sigma_{XX}}.
    
    References
    ----------
    - Geweke (1982); Barnett and Seth (2014, 2015).
    """
    # Compare full and reduced innovation covariance volumes to obtain Geweke time-domain Granger causality.
    return state_space_temporal_mvgc(
        as_innovations(model),
        source=source,
        target=target,
        conditional=conditional,
        base=base,
    )


def spectral_mvgc(
    model: Model,
    source,
    target,
    frequencies: torch.Tensor,
    *,
    conditional=(),
    base: float = math.e,
) -> torch.Tensor:
    """Compute conditional spectral multivariate Granger causality.
    
    The frequency-resolved decomposition is obtained from innovations-form transfer
    functions and integrates to temporal GC.
    
    References
    ----------
    - Geweke (1982); Barnett and Seth (2014, 2015).
    """
    # Decompose the predictive covariance ratio over frequency using the model transfer function and spectrum.
    return state_space_spectral_mvgc(
        as_innovations(model),
        source=source,
        target=target,
        frequencies=frequencies,
        conditional=conditional,
        base=base,
    )


def _transition_radius(model: Model) -> torch.Tensor:
    """Transition radius.
    
    Parameters
    ----------
    model
        VAR or linear state-space model.
    
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
    transition = as_innovations(model).transition
    # Companion eigenvalues determine stationarity through their spectral radius.
    # Obtain dynamical modes whose moduli determine stability and characteristic decay.
    return torch.linalg.eigvals(transition).abs().amax(-1)


def _criticality(model: Model, context: ModelMeasureContext) -> dict[str, torch.Tensor]:
    """Criticality.
    
    Parameters
    ----------
    model
        VAR or linear state-space model.
    context
        Optional precomputed model-measure context.
    
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
    radius = _transition_radius(model)
    tiny = torch.finfo(radius.dtype).tiny
    safe = radius.clamp(min=tiny, max=1.0 - torch.finfo(radius.dtype).eps)
    result = {
        "spectral_radius": radius,
        "stability_margin": 1.0 - radius,
        "dominant_timescale": torch.where(
            radius <= tiny, torch.zeros_like(radius), -1.0 / torch.log(safe)
        ),
    }
    if context.observation_covariance is not None:
        result["covariance_amplification"] = covariance_amplification_from_model(
            model,
            observation_covariance=context.observation_covariance,
        )
    return result


def compute_all_model_measures(
    model: Model,
    config: ModelMeasureConfig | None = None,
    *,
    context: ModelMeasureContext | None = None,
) -> dict[str, Any]:
    """Compute primary measures using the most appropriate canonical form."""
    config = ModelMeasureConfig() if config is None else config
    context = build_measure_context(model, config) if context is None else context
    if context.model is not model:
        raise ValueError("context was built for a different model")

    result: dict[str, Any] = {
        "model_type": type(model).__name__,
        "context": context,
        "available": [],
        "not_available": {},
    }

    result["dynamics"] = {
        "entropy_rate": entropy_rate_from_model(model, base=config.base),
    }
    result["available"].append("entropy_rate")
    result["criticality"] = _criticality(model, context)
    result["available"].append("criticality")

    if context.cross_spectral_density is not None:
        result["frequency"] = {
            "cross_spectral_density": context.cross_spectral_density,
            "spectral_entropy": spectral_entropy_from_spectrum(
                context.cross_spectral_density
            ),
        }
        result["available"].extend(["cross_spectral_density", "spectral_entropy"])

    if isinstance(model, (VARSystem, StateSpaceModel)):
        result["gaussian"] = gaussian_measures_from_model(
            model,
            base=config.base,
            covariance=context.observation_covariance,
        )
        result["autocovariances"] = context.autocovariances
        result["dynamics"].update(
            {
                "predictive_information": predictive_information_from_model(
                    model,
                    observation_covariance=context.observation_covariance,
                    base=config.base,
                ),
                "active_information_storage": finite_lag_ais(
                    context.autocovariances,
                    config.ais_lag,
                    base=config.base,
                ),
                "active_information_storage_lag": config.ais_lag,
            }
        )
        result["cmem"] = cmem_from_primitives(
            context.observation_covariance,
            context.innovations.innovation_covariance,
            context.autocovariances,
            curve_max_lag=config.cmem_max_lag,
            decomposition_max_lag=config.cmem_decomposition_max_lag,
        )
        result["available"].extend(
            ["gaussian", "autocovariances", "predictive_information", "active_information_storage", "cmem"]
        )
        if config.phiid_variables is not None:
            result["phiid"] = phiid_from_model(
                model,
                config.phiid_variables,
                lag=config.phiid_lag,
                autocovariance_sequence=context.autocovariances,
            )
            result["available"].append("phiid")
        if config.macro_projection is not None:
            result["emergence"] = emergence_from_model(
                model,
                config.macro_projection,
                observation_covariance=context.observation_covariance,
                base=config.base,
            )
            result["available"].append("emergence")
    else:
        reason = "stationary observation covariance is not stored by InnovationsStateSpace"
        for name in ("gaussian", "autocovariances", "predictive_information", "active_information_storage", "cmem", "phiid", "emergence"):
            result["not_available"][name] = reason

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
                frequencies=config.frequencies,
                conditional=config.conditional,
                base=config.base,
            )
        result["available"].append("mvgc")

    if isinstance(model, StateSpaceModel):
        result["control"] = {
            "dynamical_dependence": dynamical_dependence(model, base=config.base),
        }
        if config.partition is not None:
            result["control"]["stochastic_interaction"] = stochastic_interaction(
                model,
                config.partition,
                base=config.base,
            )
        result["available"].append("control")
    else:
        result["not_available"]["control"] = (
            "dynamical dependence requires an explicit latent StateSpaceModel"
        )

    result["available"] = tuple(result["available"])
    return result


__all__ = [
    "Model", "CovarianceModel", "ModelMeasureConfig", "ModelMeasureContext",
    "model_autocovariances", "required_autocovariance_max_lag",
    "build_measure_context", "stationary_observation_covariance",
    "pairwise_gaussian_mutual_information", "gaussian_measures_from_model",
    "past_future_covariance", "phiid_from_model", "temporal_mvgc",
    "spectral_mvgc", "compute_all_model_measures",
]
