r"""Measure registry for one shared resampling ensemble.

The confidence layer delegates scientific calculations to the canonical
:func:`complextorch.compute_all_model_measures` aggregate. This module only
adapts inference-specific legacy configuration and aliases; it does not maintain
an alternative implementation of model-derived measures.

References
----------
- Beda, A., Simpson, D. M., and Faes, L. (2017). Estimation of confidence
  limits for descriptive indexes derived from autoregressive analysis of time
  series. *PLoS ONE*, 12(10), e0186694.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import torch

from .measures.primary import ModelMeasureConfig, compute_all_model_measures
from .measures.rates import (
    gaussian_instantaneous_information_rate,
    gaussian_mutual_information_rate,
    gaussian_transfer_entropy_rate,
    spectral_gaussian_mutual_information_rate,
    spectral_gaussian_transfer_entropy_rate,
)
from .transformations import ModelSystem

GroupInput = int | Sequence[int]


@dataclass(frozen=True)
class InferenceMeasureConfig:
    """Backward-compatible configuration for confidence-interval measures.

    ``primary`` is the source of truth. Historical inference-only OIR and HOP
    fields are translated into the corresponding :class:`ModelMeasureConfig`
    fields before the canonical aggregate is evaluated. They are retained to
    avoid breaking the public API.
    """

    primary: ModelMeasureConfig = field(default_factory=ModelMeasureConfig)
    oir_groups: Sequence[GroupInput] | None = None
    delta_oir_target_group: int | None = None
    hop_sources: Sequence[GroupInput] | None = None
    hop_target: GroupInput | None = None
    half_open: bool = False

    def __post_init__(self) -> None:
        """Validate coupled legacy HOP configuration fields."""
        if (self.hop_sources is None) != (self.hop_target is None):
            raise ValueError("hop_sources and hop_target must be supplied together")


def _groups(groups: Sequence[GroupInput] | None):
    """Normalize integer or sequence group specifications to tuples."""
    if groups is None:
        return None
    return tuple((group,) if isinstance(group, int) else tuple(group) for group in groups)


def _primary_config(config: InferenceMeasureConfig) -> ModelMeasureConfig:
    """Map legacy inference fields onto the canonical measure configuration."""
    updates: dict[str, Any] = {}
    if config.oir_groups is not None:
        updates["rate_groups"] = _groups(config.oir_groups)
    if config.hop_sources is not None:
        updates.update(
            hop_sources=_groups(config.hop_sources),
            hop_target=(
                (config.hop_target,)
                if isinstance(config.hop_target, int)
                else tuple(config.hop_target)
            ),
            hop_half_open=config.half_open,
        )
    elif config.half_open:
        updates["hop_half_open"] = True
    return replace(config.primary, **updates) if updates else config.primary


def _result_tensors(result: Any) -> dict[str, torch.Tensor]:
    """Expose tensor fields of a decomposition result without recomputation."""
    output: dict[str, torch.Tensor] = {}
    for name in (
        "subset_mir", "subset_gc", "redundancy", "atoms", "unique",
        "redundant", "synergistic", "delta",
    ):
        value = getattr(result, name, None)
        if torch.is_tensor(value):
            output[name] = value
    return output


def evaluate_resampling_measures(
    system: ModelSystem,
    config: InferenceMeasureConfig,
) -> dict[str, Any]:
    """Evaluate all compatible measures once on a canonical model batch."""
    primary = _primary_config(config)
    result: dict[str, Any] = dict(compute_all_model_measures(system, primary))
    innovations = result["primitives"]["innovations_state_space"]
    rates = result.setdefault("rates", {})

    # Historical names are aliases to canonical aggregate outputs.
    if "o_information" in rates:
        rates["o_information_rate"] = rates["o_information"]
    if config.delta_oir_target_group is not None:
        target = config.delta_oir_target_group
        if "delta_o_information" not in rates:
            raise ValueError("delta O-information rate requires at least three groups")
        rates["delta_o_information_rate"] = rates["delta_o_information"][..., target]
        if "spectral_delta_o_information" in rates:
            rates["spectral_delta_o_information_rate"] = rates[
                "spectral_delta_o_information"
            ][..., target, :]

    # Group-specific source/target rates are not part of the always-computed
    # pairwise aggregate, so evaluate only these requested quantities here.
    if primary.source is not None and primary.target is not None:
        source, target = primary.source, primary.target
        rates.update(
            mutual_information_rate=gaussian_mutual_information_rate(
                innovations, source, target, base=primary.base
            ),
            transfer_entropy_rate=gaussian_transfer_entropy_rate(
                innovations, source, target, base=primary.base
            ),
            instantaneous_information_rate=gaussian_instantaneous_information_rate(
                innovations, source, target, base=primary.base
            ),
        )
        if primary.frequencies is not None:
            rates.update(
                spectral_mutual_information_rate=spectral_gaussian_mutual_information_rate(
                    innovations,
                    source,
                    target,
                    primary.frequencies,
                    sampling_frequency=primary.sampling_frequency,
                    base=primary.base,
                ),
                spectral_transfer_entropy_rate=spectral_gaussian_transfer_entropy_rate(
                    innovations,
                    source,
                    target,
                    primary.frequencies,
                    sampling_frequency=primary.sampling_frequency,
                    base=primary.base,
                ),
            )

    # The canonical aggregate returns typed HOP/PIRD/PDGC results. Preserve the
    # historical inference mapping shape by exposing references to those tensors.
    if "pird" in result and "pdgc" in result:
        pird = _result_tensors(result["pird"])
        pdgc = _result_tensors(result["pdgc"])
        spectral_pird = _result_tensors(result["spectral_pird"])
        spectral_pdgc = _result_tensors(result["spectral_pdgc"])
        result["pird"] = pird
        result["pdgc"] = pdgc
        result["spectral_pird"] = spectral_pird
        result["spectral_pdgc"] = spectral_pdgc
        result["hop"] = {"pird": pird, "pdgc": pdgc}
        result["spectral_hop"] = {
            "pird": spectral_pird,
            "pdgc": spectral_pdgc,
        }
    return result


__all__ = ["InferenceMeasureConfig", "evaluate_resampling_measures"]
