"""Strict analytical measures computed from supplied generative models.

Primary functions do not refit observations. A shared context computes the
maximum required autocovariance lag and caches reusable model-derived
primitives. :func:`compute_all_model_measures` is the canonical aggregate
entry point and returns every primary measure that is applicable without
additional observations; measures requiring structural choices are controlled
by :class:`ModelMeasureConfig`.

References
----------
- Cover, T. M. and Thomas, J. A. (2006).
- Barnett, L. and Seth, A. K. (2014, 2015).
- Faes, L. et al. (2022).
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
from .mvgc import (
    pairwise_temporal_mvgc,
    state_space_spectral_mvgc,
    state_space_temporal_mvgc,
)
from .oir import o_information_rate, spectral_o_information_rate
from .phid import gaussian_phiid_atoms
from .rates import (
    gaussian_instantaneous_information_rate,
    gaussian_mutual_information_rate,
    spectral_gaussian_mutual_information_rate,
    spectral_gaussian_transfer_entropy_rate,
)


@dataclass(frozen=True)
class ModelMeasureConfig:
    """Structural choices for model-derived analytical measures.

    Parameters
    ----------
    frequencies
        Optional frequency grid. Supplying it enables all frequency-resolved
        primary families that are otherwise applicable.
    sampling_frequency
        Sampling frequency associated with ``frequencies``.
    autocovariance_max_lag, ais_lag, cmem_max_lag, cmem_decomposition_max_lag
        Finite-lag requirements resolved into one shared autocovariance cache.
    source, target, conditional
        Optional source/target grouping for a specifically requested MVGC in
        addition to the always-computed singleton pairwise MVGC matrix.
    phiid_variables, phiid_lag
        Optional bivariate PhiID selection.
    phiid_redundancies
        PhiID redundancy backends evaluated for the selected bivariate process.
        The default computes MMI, CCS, I_dep-a, and I_dep-b from the same cached
        past/future covariance.
    phiid_ccs_qmc_samples
        Deterministic Sobol node count for the Gaussian CCS expectation.
    macro_projection
        Optional coarse-graining used by emergence and dynamical dependence.
    partition
        Optional partition for stochastic interaction.
    rate_groups
        Optional grouping for O-information rate. ``None`` uses every observed
        channel as a singleton group.
    hop_sources, hop_target
        Optional two- or three-source grouping and target used jointly by PIRD,
        PDGC, and HOP. Both must be supplied together.
    hop_half_open
        Use the exact Faes/HOP half-open integration convention for integrated
        PIRD/PDGC/HOP quantities.
    base
        Logarithm base used by information quantities.
    """

    frequencies: torch.Tensor | None = None
    sampling_frequency: float = 1.0
    autocovariance_max_lag: int = 1
    ais_lag: int = 1
    cmem_max_lag: int = 1
    cmem_decomposition_max_lag: int = 1
    source: tuple[int, ...] | None = None
    target: tuple[int, ...] | None = None
    conditional: tuple[int, ...] | None = None
    phiid_variables: tuple[int, int] | None = None
    phiid_lag: int = 1
    phiid_redundancies: tuple[str, ...] = ("mmi", "ccs", "idep_a", "idep_b")
    phiid_ccs_qmc_samples: int = 4096
    macro_projection: torch.Tensor | None = None
    partition: tuple[tuple[int, ...], ...] | None = None
    rate_groups: tuple[tuple[int, ...], ...] | None = None
    hop_sources: tuple[tuple[int, ...], ...] | None = None
    hop_target: tuple[int, ...] | None = None
    hop_half_open: bool = False
    base: float = 2.0

    def __post_init__(self) -> None:
        """Validate finite-lag, spectral, PhiID, and HOP settings."""
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
        allowed_redundancies = {"mmi", "ccs", "idep_a", "idep_b"}
        if not self.phiid_redundancies:
            raise ValueError("phiid_redundancies must contain at least one backend")
        if any(name not in allowed_redundancies for name in self.phiid_redundancies):
            raise ValueError(
                "phiid_redundancies entries must be one of mmi, ccs, idep_a, idep_b"
            )
        if len(set(self.phiid_redundancies)) != len(self.phiid_redundancies):
            raise ValueError("phiid_redundancies must not contain duplicates")
        if self.phiid_ccs_qmc_samples < 32:
            raise ValueError("phiid_ccs_qmc_samples must be at least 32")
        if (self.hop_sources is None) != (self.hop_target is None):
            raise ValueError("hop_sources and hop_target must be supplied together")
        if self.hop_sources is not None and len(self.hop_sources) not in (2, 3):
            raise ValueError("hop_sources must contain exactly two or three source groups")


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
        """Return the largest cached non-negative autocovariance lag."""
        return 0 if self.autocovariances is None else int(self.autocovariances.shape[-3] - 1)


def required_autocovariance_max_lag(model: Model, config: ModelMeasureConfig) -> int:
    """Resolve all finite-delay families to one shared maximum lag."""
    required = [
        config.autocovariance_max_lag,
        config.ais_lag,
        config.cmem_max_lag,
        config.cmem_decomposition_max_lag,
    ]
    if isinstance(model, VARSystem):
        # CMem totals/decomposition use the complete VAR Markov history.
        required.append(model.order)
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
    """Return the stationary observation covariance implied by ``model``."""
    return observation_autocovariances(model, 0)[..., 0, :, :]


def model_autocovariances(model: CovarianceModel, max_lag: int = 1) -> torch.Tensor:
    """Return model-implied autocovariances through ``max_lag``."""
    return observation_autocovariances(model, max_lag)


def pairwise_gaussian_mutual_information(
    covariance: torch.Tensor, *, base: float = 2.0
) -> torch.Tensor:
    """Return the symmetric singleton-pair Gaussian MI matrix."""
    covariance = torch.as_tensor(covariance)
    n_variables = covariance.shape[-1]
    result = torch.zeros(
        (*covariance.shape[:-2], n_variables, n_variables),
        dtype=covariance.dtype,
        device=covariance.device,
    )
    for left in range(n_variables):
        for right in range(left + 1, n_variables):
            index = torch.as_tensor([left, right], dtype=torch.long, device=covariance.device)
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
    """Return static Gaussian measures from the model covariance."""
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
    """Build the bivariate past/future covariance used by Gaussian PhiID."""
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
    """Return Gaussian PhiID atoms from model-implied lagged covariance."""
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
    conditional=None,
    base: float = math.e,
) -> torch.Tensor:
    """Compute conditional time-domain multivariate Granger causality."""
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
    conditional=None,
    base: float = math.e,
) -> torch.Tensor:
    """Compute conditional spectral multivariate Granger causality."""
    return state_space_spectral_mvgc(
        as_innovations(model),
        source=source,
        target=target,
        frequencies=frequencies,
        conditional=conditional,
        base=base,
    )


def _transition_radius(model: Model) -> torch.Tensor:
    """Return the spectral radius of the canonical transition matrix."""
    transition = as_innovations(model).transition
    return torch.linalg.eigvals(transition).abs().amax(-1)


def _criticality(model: Model, context: ModelMeasureContext) -> dict[str, torch.Tensor]:
    """Compute model-derived criticality measures."""
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


def _pairwise_scalar_matrix(
    system: InnovationsStateSpace,
    measure,
    *,
    directed: bool,
    base: float,
) -> torch.Tensor:
    """Evaluate one singleton-pair scalar measure while batching model axes."""
    n_variables = system.observation.shape[-2]
    covariance = system.innovation_covariance
    result = torch.zeros(
        (*covariance.shape[:-2], n_variables, n_variables),
        dtype=covariance.dtype,
        device=covariance.device,
    )
    for left in range(n_variables):
        right_range = range(n_variables) if directed else range(left + 1, n_variables)
        for right in right_range:
            if left == right:
                continue
            value = measure(system, (left,), (right,), base=base)
            result[..., left, right] = value
            if not directed:
                result[..., right, left] = value
    return result


def _pairwise_spectral_matrix(
    system: InnovationsStateSpace,
    frequencies: torch.Tensor,
    measure,
    *,
    directed: bool,
    sampling_frequency: float,
    base: float,
) -> torch.Tensor:
    """Evaluate one singleton-pair spectral measure over a model batch."""
    n_variables = system.observation.shape[-2]
    frequency = torch.as_tensor(
        frequencies, dtype=system.transition.dtype, device=system.transition.device
    )
    covariance = system.innovation_covariance
    result = torch.zeros(
        (*covariance.shape[:-2], n_variables, n_variables, frequency.numel()),
        dtype=covariance.dtype,
        device=covariance.device,
    )
    for left in range(n_variables):
        right_range = range(n_variables) if directed else range(left + 1, n_variables)
        for right in right_range:
            if left == right:
                continue
            value = measure(
                system,
                (left,),
                (right,),
                frequency,
                sampling_frequency=sampling_frequency,
                base=base,
            )
            result[..., left, right, :] = value
            if not directed:
                result[..., right, left, :] = value
    return result


def _pairwise_spectral_mvgc(
    model: Model,
    frequencies: torch.Tensor,
    *,
    base: float,
) -> torch.Tensor:
    """Return all ordered singleton-to-singleton spectral MVGC curves."""
    innovations = as_innovations(model)
    n_variables = innovations.observation.shape[-2]
    frequency = torch.as_tensor(
        frequencies, dtype=innovations.transition.dtype, device=innovations.transition.device
    )
    covariance = innovations.innovation_covariance
    result = torch.zeros(
        (*covariance.shape[:-2], n_variables, n_variables, frequency.numel()),
        dtype=covariance.dtype,
        device=covariance.device,
    )
    for source in range(n_variables):
        for target in range(n_variables):
            if source == target:
                continue
            result[..., source, target, :] = state_space_spectral_mvgc(
                innovations,
                source=(source,),
                target=(target,),
                frequencies=frequency,
                base=base,
            )
    return result


def _mean_unique_pairs(matrix: torch.Tensor) -> torch.Tensor:
    """Average the strict upper triangle of a symmetric pairwise matrix."""
    n_variables = matrix.shape[-1]
    indices = torch.triu_indices(
        n_variables, n_variables, offset=1, device=matrix.device
    )
    return matrix[..., indices[0], indices[1]].mean(dim=-1)


def compute_all_model_measures(
    model: Model,
    config: ModelMeasureConfig | None = None,
    *,
    context: ModelMeasureContext | None = None,
) -> dict[str, Any]:
    """Compute every applicable primary measure from one canonical model.

    Shared canonical primitives are built once in ``context`` and exposed in
    ``result["primitives"]``. Singleton pairwise information-rate and MVGC
    matrices are computed whenever at least two observation variables exist.
    Optional PhiID, emergence, stochastic interaction, HOP/PIRD/PDGC, and
    specifically grouped MVGC families are evaluated when their structural
    configuration is supplied. No quantity is estimated from observations.
    """
    config = ModelMeasureConfig() if config is None else config
    context = build_measure_context(model, config) if context is None else context
    if context.model is not model:
        raise ValueError("context was built for a different model")

    result: dict[str, Any] = {
        "model_type": type(model).__name__,
        "context": context,
        "primitives": {
            "innovations_state_space": context.innovations,
            "observation_covariance": context.observation_covariance,
            "autocovariances": context.autocovariances,
            "cross_spectral_density": context.cross_spectral_density,
        },
        "available": ["primitives"],
        "not_available": {},
    }

    result["dynamics"] = {
        "entropy_rate": entropy_rate_from_model(model, base=config.base),
    }
    result["available"].append("entropy_rate")
    result["criticality"] = _criticality(model, context)
    result["available"].append("criticality")

    pairwise_temporal = pairwise_temporal_mvgc(context.innovations, base=config.base)
    result["mvgc"] = {"pairwise_temporal": pairwise_temporal}
    result["available"].append("mvgc")

    innovations = context.innovations
    n_observations = innovations.observation.shape[-2]
    pairwise_mir = _pairwise_scalar_matrix(
        innovations,
        gaussian_mutual_information_rate,
        directed=False,
        base=config.base,
    )
    pairwise_instantaneous = _pairwise_scalar_matrix(
        innovations,
        gaussian_instantaneous_information_rate,
        directed=False,
        base=config.base,
    )
    rates: dict[str, torch.Tensor] = {
        "pairwise_mutual_information": pairwise_mir,
        "pairwise_transfer_entropy": 0.5 * pairwise_temporal,
        "pairwise_instantaneous_information": pairwise_instantaneous,
    }
    if n_observations >= 2:
        rates["mean_pairwise_mutual_information"] = _mean_unique_pairs(pairwise_mir)
        rates["o_information"] = o_information_rate(
            innovations,
            groups=config.rate_groups,
            base=config.base,
        )
    else:
        reason = "requires at least two observation variables"
        result["not_available"]["mean_pairwise_mutual_information_rate"] = reason
        result["not_available"]["o_information_rate"] = reason
    result["rates"] = rates
    result["available"].append("rates")

    if context.cross_spectral_density is not None:
        result["frequency"] = {
            "cross_spectral_density": context.cross_spectral_density,
            "spectral_entropy": spectral_entropy_from_spectrum(
                context.cross_spectral_density
            ),
        }
        result["mvgc"]["pairwise_spectral"] = _pairwise_spectral_mvgc(
            model,
            config.frequencies,
            base=config.base,
        )
        result["rates"].update(
            {
                "pairwise_spectral_mutual_information": _pairwise_spectral_matrix(
                    innovations,
                    config.frequencies,
                    spectral_gaussian_mutual_information_rate,
                    directed=False,
                    sampling_frequency=config.sampling_frequency,
                    base=config.base,
                ),
                "pairwise_spectral_transfer_entropy": _pairwise_spectral_matrix(
                    innovations,
                    config.frequencies,
                    spectral_gaussian_transfer_entropy_rate,
                    directed=True,
                    sampling_frequency=config.sampling_frequency,
                    base=config.base,
                ),
            }
        )
        if n_observations >= 2:
            result["rates"]["spectral_o_information"] = spectral_o_information_rate(
                innovations,
                config.frequencies,
                groups=config.rate_groups,
                sampling_frequency=config.sampling_frequency,
                base=config.base,
            )
        else:
            result["not_available"]["spectral_o_information_rate"] = (
                "requires at least two observation variables"
            )
        result["available"].extend(["cross_spectral_density", "spectral_entropy"])

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
        decomposition_max_lag=(
            model.order if isinstance(model, VARSystem)
            else config.cmem_decomposition_max_lag
        ),
    )
    result["available"].extend(
        [
            "gaussian",
            "autocovariances",
            "predictive_information",
            "active_information_storage",
            "cmem",
        ]
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

    if config.source is not None and config.target is not None:
        result["mvgc"]["temporal"] = temporal_mvgc(
            model,
            source=config.source,
            target=config.target,
            conditional=config.conditional,
            base=config.base,
        )
        if config.frequencies is not None:
            result["mvgc"]["spectral"] = spectral_mvgc(
                model,
                source=config.source,
                target=config.target,
                frequencies=config.frequencies,
                conditional=config.conditional,
                base=config.base,
            )

    control: dict[str, torch.Tensor] = {}
    if config.macro_projection is not None:
        control["dynamical_dependence"] = dynamical_dependence(
            model,
            config.macro_projection,
            base=config.base,
        )
    else:
        result["not_available"]["dynamical_dependence"] = (
            "dynamical dependence requires macro_projection"
        )
    if isinstance(model, StateSpaceModel) and config.partition is not None:
        control["stochastic_interaction"] = stochastic_interaction(
            model,
            config.partition,
            base=config.base,
        )
    if control:
        result["control"] = control
        result["available"].append("control")
    elif "dynamical_dependence" not in result["not_available"]:
        result["not_available"]["control"] = "no requested control measures are available"

    from ._aggregate import extend_model_measure_result

    result = extend_model_measure_result(model, config, context, result)
    result["available"] = tuple(result["available"])
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
