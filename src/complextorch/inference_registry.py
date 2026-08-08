r"""Registry of model-derived measures evaluated on one resampling ensemble.

The registry is deliberately separate from the resampling engine. It consumes a
single canonical :class:`~complextorch.VARSystem` (possibly batched over bootstrap
replicates) and evaluates every configured compatible analytical measure without
refitting observations. High-order HOP outputs reuse the already-computed PIRD
and PDGC result tensors rather than calling the HOP composition a second time.

References
----------
- Beda, A., Simpson, D. M., and Faes, L. (2017). Estimation of confidence
  limits for descriptive indexes derived from autoregressive analysis of time
  series. *PLoS ONE*, 12(10), e0186694.
- Faes, L. et al. (2022). A new framework for the time- and frequency-domain
  assessment of high-order interactions in networks of random processes.
  *IEEE Transactions on Signal Processing*, 70, 5766-5777.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from .control import var_to_innovations_state_space
from .measures.oir import (
    delta_o_information_rate,
    o_information_rate,
    spectral_delta_o_information_rate,
    spectral_o_information_rate,
)
from .measures.pdgc import (
    partial_granger_causality_decomposition,
    spectral_partial_granger_causality_decomposition,
)
from .measures.pird import (
    partial_information_rate_decomposition,
    spectral_partial_information_rate_decomposition,
)
from .measures.primary import ModelMeasureConfig, compute_all_model_measures
from .measures.rates import (
    gaussian_instantaneous_information_rate,
    gaussian_mutual_information_rate,
    gaussian_transfer_entropy_rate,
    spectral_gaussian_mutual_information_rate,
    spectral_gaussian_transfer_entropy_rate,
)
from .representations import VARSystem

GroupInput = int | Sequence[int]


@dataclass(frozen=True)
class InferenceMeasureConfig:
    """Measure configuration for one shared VAR-resampling ensemble.

    Parameters
    ----------
    primary
        Existing ComplexTorch primary-measure configuration. Its ``source`` and
        ``target`` are reused for MIR, TE, instantaneous information rate and
        MVGC so the same directed partition is evaluated consistently.
    oir_groups
        Optional O-information-rate process groups. ``None`` uses every channel
        as a singleton group, matching :func:`o_information_rate`.
    delta_oir_target_group
        Optional zero-based group position for delta O-information rate. It is
        evaluated only when supplied.
    hop_sources, hop_target
        Exactly two or three source groups plus one target group for PIRD/PDGC/
        HOP. When supplied, PIRD and PDGC are each computed once; HOP paths alias
        those same result tensors and introduce no duplicate numerical work.
    half_open
        Use the Faes/HOP half-open spectral integration convention for temporal
        PIRD/PDGC when true.
    """

    primary: ModelMeasureConfig = field(default_factory=ModelMeasureConfig)
    oir_groups: Sequence[GroupInput] | None = None
    delta_oir_target_group: int | None = None
    hop_sources: Sequence[GroupInput] | None = None
    hop_target: GroupInput | None = None
    half_open: bool = False

    def __post_init__(self) -> None:
        """Validate coupled high-order configuration fields."""
        if (self.hop_sources is None) != (self.hop_target is None):
            raise ValueError("hop_sources and hop_target must be supplied together")


def _result_tensors(result: Any) -> dict[str, torch.Tensor]:
    """Extract only tensor-valued scientific fields from a result dataclass."""
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
    system: VARSystem,
    config: InferenceMeasureConfig,
) -> dict[str, Any]:
    """Evaluate all configured measures once on a supplied VARSystem batch.

    No observations are fitted here. The leading model batch axis is preserved
    by every registered consumer and is therefore the bootstrap axis used by the
    confidence-interval engine.
    """
    result: dict[str, Any] = {
        "primary": compute_all_model_measures(system, config.primary),
    }
    innovations = var_to_innovations_state_space(system)
    base = config.primary.base
    frequencies = config.primary.frequencies
    sampling_frequency = config.primary.sampling_frequency

    result["rates"] = {
        "o_information": o_information_rate(
            innovations, groups=config.oir_groups, base=base
        )
    }
    if frequencies is not None:
        result["rates"]["spectral_o_information"] = spectral_o_information_rate(
            innovations,
            frequencies,
            groups=config.oir_groups,
            sampling_frequency=sampling_frequency,
            base=base,
        )
    if config.delta_oir_target_group is not None:
        result["rates"]["delta_o_information"] = delta_o_information_rate(
            innovations,
            config.delta_oir_target_group,
            groups=config.oir_groups,
            base=base,
        )
        if frequencies is not None:
            result["rates"]["spectral_delta_o_information"] = (
                spectral_delta_o_information_rate(
                    innovations,
                    frequencies,
                    config.delta_oir_target_group,
                    groups=config.oir_groups,
                    sampling_frequency=sampling_frequency,
                    base=base,
                )
            )

    if config.primary.source is not None and config.primary.target is not None:
        source = config.primary.source
        target = config.primary.target
        result["rates"].update(
            {
                "mutual_information": gaussian_mutual_information_rate(
                    innovations, source, target, base=base
                ),
                "transfer_entropy": gaussian_transfer_entropy_rate(
                    innovations, source, target, base=base
                ),
                "instantaneous_information": gaussian_instantaneous_information_rate(
                    innovations, source, target, base=base
                ),
            }
        )
        if frequencies is not None:
            result["rates"].update(
                {
                    "spectral_mutual_information": (
                        spectral_gaussian_mutual_information_rate(
                            innovations,
                            source,
                            target,
                            frequencies,
                            sampling_frequency=sampling_frequency,
                            base=base,
                        )
                    ),
                    "spectral_transfer_entropy": (
                        spectral_gaussian_transfer_entropy_rate(
                            innovations,
                            source,
                            target,
                            frequencies,
                            sampling_frequency=sampling_frequency,
                            base=base,
                        )
                    ),
                }
            )

    if config.hop_sources is not None and config.hop_target is not None:
        if frequencies is None:
            raise ValueError("PIRD/PDGC/HOP confidence intervals require frequencies")
        pird = partial_information_rate_decomposition(
            innovations,
            config.hop_sources,
            config.hop_target,
            frequencies,
            sampling_frequency=sampling_frequency,
            base=base,
            half_open=config.half_open,
        )
        pdgc = partial_granger_causality_decomposition(
            innovations,
            config.hop_sources,
            config.hop_target,
            frequencies,
            sampling_frequency=sampling_frequency,
            base=base,
            half_open=config.half_open,
        )
        spectral_pird = spectral_partial_information_rate_decomposition(
            innovations,
            config.hop_sources,
            config.hop_target,
            frequencies,
            sampling_frequency=sampling_frequency,
            base=base,
        )
        spectral_pdgc = spectral_partial_granger_causality_decomposition(
            innovations,
            config.hop_sources,
            config.hop_target,
            frequencies,
            sampling_frequency=sampling_frequency,
            base=base,
        )
        pird_tensors = _result_tensors(pird)
        pdgc_tensors = _result_tensors(pdgc)
        spectral_pird_tensors = _result_tensors(spectral_pird)
        spectral_pdgc_tensors = _result_tensors(spectral_pdgc)
        result["pird"] = pird_tensors
        result["pdgc"] = pdgc_tensors
        result["spectral_pird"] = spectral_pird_tensors
        result["spectral_pdgc"] = spectral_pdgc_tensors
        # HOP is composition, not a third numerical estimator. Reuse references.
        result["hop"] = {"pird": pird_tensors, "pdgc": pdgc_tensors}
        result["spectral_hop"] = {
            "pird": spectral_pird_tensors,
            "pdgc": spectral_pdgc_tensors,
        }
    return result


__all__ = ["InferenceMeasureConfig", "evaluate_resampling_measures"]
