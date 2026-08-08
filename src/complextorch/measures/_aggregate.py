"""Private composition helpers for the complete model-measure aggregate API.

This module contains no estimators and introduces no alternative numerical
kernels. It composes already validated primary measures from the canonical
model primitives supplied by :class:`ModelMeasureContext`.
"""
from __future__ import annotations

from typing import Any

import torch

from ..control import innovations_transfer_function, stochastic_interaction
from ..spectra import integrate_spectral_rate
from .entropy_rate import marginal_entropy_rate, spectral_entropy_rate
from .hop import HOPResult, SpectralHOPResult
from .oir import delta_o_information_rate, spectral_delta_o_information_rate
from .pdgc import PDGCResult
from .phid import gaussian_phiid_atoms
from .pird import PIRDResult


def _append_available(result: dict[str, Any], name: str) -> None:
    """Append one availability label without introducing duplicates."""
    available = result["available"]
    if isinstance(available, tuple):
        available = list(available)
        result["available"] = available
    if name not in available:
        available.append(name)


def _past_future_covariance_from_cache(
    autocovariances: torch.Tensor,
    variables: tuple[int, int],
    lag: int,
) -> torch.Tensor:
    """Build one bivariate past/future covariance from the shared lag cache."""
    if autocovariances.shape[-3] <= lag:
        raise ValueError("autocovariances do not contain the requested PhiID lag")
    gamma = autocovariances
    single = gamma.ndim == 3
    if single:
        gamma = gamma.unsqueeze(0)
    index = torch.as_tensor(variables, dtype=torch.long, device=gamma.device)
    present = gamma[:, 0].index_select(-2, index).index_select(-1, index)
    future_past = gamma[:, lag].index_select(-2, index).index_select(-1, index)
    top = torch.cat([present, future_past.transpose(-1, -2)], dim=-1)
    bottom = torch.cat([future_past, present], dim=-1)
    covariance = torch.cat([top, bottom], dim=-2)
    return covariance[0] if single else covariance


def _integrate_spectral_hop(
    spectral: SpectralHOPResult,
    *,
    sampling_frequency: float,
    half_open: bool,
) -> HOPResult:
    """Integrate one already-computed spectral HOP result without recomputation."""
    def integrate(value: torch.Tensor) -> torch.Tensor:
        """Integrate a HOP component over its final frequency axis."""
        return integrate_spectral_rate(
            value,
            spectral.frequencies,
            sampling_frequency=sampling_frequency,
            half_open=half_open,
        )

    spird = spectral.pird
    pird = PIRDResult(
        sources=spird.sources,
        target=spird.target,
        source_subsets=spird.source_subsets,
        antichains=spird.antichains,
        subset_mir=integrate(spird.subset_mir),
        redundancy=integrate(spird.redundancy),
        atoms=integrate(spird.atoms),
        unique=integrate(spird.unique),
        redundant=integrate(spird.redundant),
        synergistic=integrate(spird.synergistic),
        delta=integrate(spird.delta),
    )
    spdgc = spectral.pdgc
    pdgc = PDGCResult(
        sources=spdgc.sources,
        target=spdgc.target,
        source_subsets=spdgc.source_subsets,
        antichains=spdgc.antichains,
        subset_gc=integrate(spdgc.subset_gc),
        redundancy=integrate(spdgc.redundancy),
        atoms=integrate(spdgc.atoms),
        unique=integrate(spdgc.unique),
        redundant=integrate(spdgc.redundant),
        synergistic=integrate(spdgc.synergistic),
        delta=integrate(spdgc.delta),
    )
    return HOPResult(
        sources=spectral.sources,
        target=spectral.target,
        pird=pird,
        pdgc=pdgc,
    )


def extend_model_measure_result(
    model,
    config,
    context,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Add later primary families to the canonical aggregate result.

    The function intentionally receives the already-built context and result so
    that all additions reuse canonical innovations, autocovariances, spectra,
    and any primary quantities computed by the core aggregate path.
    """
    innovations = context.innovations
    n_observations = innovations.observation.shape[-2]

    result["dynamics"]["marginal_entropy_rate"] = marginal_entropy_rate(
        innovations,
        base=config.base,
    )
    _append_available(result, "marginal_entropy_rate")

    if config.frequencies is not None:
        frequency = torch.as_tensor(
            config.frequencies,
            dtype=innovations.transition.dtype,
            device=innovations.transition.device,
        )
        normalized_frequency = frequency / float(config.sampling_frequency)
        transfer = innovations_transfer_function(innovations, normalized_frequency)
        identity = torch.eye(
            transfer.shape[-1], dtype=transfer.dtype, device=transfer.device
        ).expand_as(transfer)
        inverse_transfer = torch.linalg.solve(transfer, identity)
        result["primitives"]["transfer_function"] = transfer
        result["primitives"]["inverse_transfer_function"] = inverse_transfer
        result["frequency"]["spectral_entropy_rate"] = spectral_entropy_rate(
            innovations,
            frequency,
            sampling_frequency=config.sampling_frequency,
            base=config.base,
        )
        _append_available(result, "spectral_entropy_rate")

    groups = (
        tuple((index,) for index in range(n_observations))
        if config.rate_groups is None
        else tuple(config.rate_groups)
    )
    if len(groups) >= 3:
        result["rates"]["delta_o_information"] = torch.stack(
            [
                delta_o_information_rate(
                    innovations,
                    target_group=index,
                    groups=groups,
                    base=config.base,
                )
                for index in range(len(groups))
            ],
            dim=-1,
        )
        if config.frequencies is not None:
            result["rates"]["spectral_delta_o_information"] = torch.stack(
                [
                    spectral_delta_o_information_rate(
                        innovations,
                        config.frequencies,
                        target_group=index,
                        groups=groups,
                        sampling_frequency=config.sampling_frequency,
                        base=config.base,
                    )
                    for index in range(len(groups))
                ],
                dim=-2,
            )
    else:
        result["not_available"]["delta_o_information_rate"] = (
            "requires at least three disjoint process groups"
        )

    if config.phiid_variables is not None and context.autocovariances is not None:
        covariance = _past_future_covariance_from_cache(
            context.autocovariances,
            config.phiid_variables,
            config.phiid_lag,
        )
        redundancies: dict[str, dict[str, torch.Tensor]] = {}
        for redundancy in config.phiid_redundancies:
            if redundancy == "mmi" and "phiid" in result:
                redundancies[redundancy] = result["phiid"]
            else:
                redundancies[redundancy] = gaussian_phiid_atoms(
                    covariance,
                    redundancy=redundancy,
                    base=config.base,
                    ccs_qmc_samples=config.phiid_ccs_qmc_samples,
                )
        result["phiid_redundancy"] = redundancies
        if "mmi" in redundancies:
            result["phiid"] = redundancies["mmi"]
        _append_available(result, "phiid_redundancy")

    hop_requested = config.hop_sources is not None or config.hop_target is not None
    if hop_requested:
        if config.hop_sources is None or config.hop_target is None:
            raise ValueError("hop_sources and hop_target must be supplied together")
        if config.frequencies is None:
            result["not_available"]["hop"] = "HOP requires a frequency grid"
        else:
            from .hop import spectral_hop_analysis

            spectral = spectral_hop_analysis(
                innovations,
                config.hop_sources,
                config.hop_target,
                config.frequencies,
                sampling_frequency=config.sampling_frequency,
                base=config.base,
            )
            integrated = _integrate_spectral_hop(
                spectral,
                sampling_frequency=config.sampling_frequency,
                half_open=config.hop_half_open,
            )
            result["hop"] = integrated
            result["spectral_hop"] = spectral
            result["pird"] = integrated.pird
            result["spectral_pird"] = spectral.pird
            result["pdgc"] = integrated.pdgc
            result["spectral_pdgc"] = spectral.pdgc
            for name in ("hop", "pird", "pdgc"):
                _append_available(result, name)
    else:
        result["not_available"]["hop"] = "requires hop_sources and hop_target"
        result["not_available"]["pird"] = "requires hop_sources and hop_target"
        result["not_available"]["pdgc"] = "requires hop_sources and hop_target"

    if config.partition is not None:
        control = result.setdefault("control", {})
        if "stochastic_interaction" not in control:
            control["stochastic_interaction"] = stochastic_interaction(
                model,
                config.partition,
                base=config.base,
            )
        _append_available(result, "control")
    else:
        result["not_available"]["stochastic_interaction"] = (
            "stochastic interaction requires partition"
        )

    return result


__all__ = ["extend_model_measure_result"]
