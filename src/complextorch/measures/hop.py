r"""High-Order analysis of random Processes (HOP) composition.

HOP is the umbrella interface collecting the two complementary high-order
process decompositions implemented by ComplexTorch:

- partial information rate decomposition (PIRD), which decomposes information
  shared dynamically between source processes and a target; and
- partial decomposition of Granger causality (PDGC), which decomposes directed
  causal influence from those sources to the target.

The HOP layer intentionally introduces no new estimator, state-space reduction,
DARE solver, spectral kernel, PID lattice, or integration rule. It calls the
validated PIRD and PDGC implementations with one common model, grouping,
frequency grid, log base, sampling frequency, and integration convention, then
checks that their structural metadata agree. Torch batch dimensions are
therefore inherited unchanged from the component decompositions.

References
----------
- Faes, L. et al. (2022). A new framework for the time- and frequency-domain
  assessment of high-order interactions in networks of random processes.
  *IEEE Transactions on Signal Processing*, 70, 5766-5777.
- Faes, L. et al. (2025). Partial information rate decomposition.
  *Physical Review Letters*, 135, 187401.
- Faes, L. et al. (2026). Dissecting spectral Granger causality through partial
  information decomposition. arXiv:2603.07634.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ..control import InnovationsStateSpace, var_to_innovations_state_space
from ..representations import VARSystem
from .oir import Group
from .pdgc import (
    PDGCResult,
    SpectralPDGCResult,
    partial_granger_causality_decomposition,
    spectral_partial_granger_causality_decomposition,
)
from .pird import (
    PIRDResult,
    SpectralPIRDResult,
    partial_information_rate_decomposition,
    spectral_partial_information_rate_decomposition,
)


@dataclass(frozen=True)
class SpectralHOPResult:
    """Matched frequency-resolved PIRD and PDGC decompositions.

    Attributes
    ----------
    frequencies
        Shared one-dimensional physical-frequency grid.
    sources
        Shared normalized source observation groups.
    target
        Shared normalized target observation group.
    pird
        Frequency-resolved partial information rate decomposition.
    pdgc
        Frequency-resolved partial decomposition of Granger causality.
    """

    frequencies: torch.Tensor
    sources: tuple[Group, ...]
    target: Group
    pird: SpectralPIRDResult
    pdgc: SpectralPDGCResult


@dataclass(frozen=True)
class HOPResult:
    """Matched integrated PIRD and PDGC decompositions.

    Attributes
    ----------
    sources
        Shared normalized source observation groups.
    target
        Shared normalized target observation group.
    pird
        Integrated partial information rate decomposition.
    pdgc
        Integrated partial decomposition of Granger causality.
    """

    sources: tuple[Group, ...]
    target: Group
    pird: PIRDResult
    pdgc: PDGCResult


def _as_innovations(system: VARSystem | InnovationsStateSpace) -> InnovationsStateSpace:
    """Return one canonical innovations model for both HOP components."""
    if isinstance(system, VARSystem):
        return var_to_innovations_state_space(system)
    if isinstance(system, InnovationsStateSpace):
        return system
    raise TypeError("system must be a VARSystem or InnovationsStateSpace")


def _validate_matched_metadata(
    pird: PIRDResult | SpectralPIRDResult,
    pdgc: PDGCResult | SpectralPDGCResult,
) -> None:
    """Guard against structural drift between the two HOP components."""
    if pird.sources != pdgc.sources or pird.target != pdgc.target:
        raise RuntimeError("PIRD and PDGC returned inconsistent source/target groups")
    if pird.source_subsets != pdgc.source_subsets:
        raise RuntimeError("PIRD and PDGC returned inconsistent source-subset ordering")
    if pird.antichains != pdgc.antichains:
        raise RuntimeError("PIRD and PDGC returned inconsistent PID-lattice ordering")
    if isinstance(pird, SpectralPIRDResult) and isinstance(pdgc, SpectralPDGCResult):
        if pird.frequencies.shape != pdgc.frequencies.shape or not torch.equal(
            pird.frequencies, pdgc.frequencies
        ):
            raise RuntimeError("PIRD and PDGC returned inconsistent frequency grids")


def spectral_hop_analysis(
    system: VARSystem | InnovationsStateSpace,
    sources: Sequence[int | Sequence[int]],
    target: int | Sequence[int],
    frequencies: torch.Tensor,
    *,
    sampling_frequency: float = 1.0,
    base: float = math.e,
) -> SpectralHOPResult:
    r"""Compute matched frequency-resolved PIRD and PDGC analyses.

    The same exact innovations model and analysis configuration are passed to
    both decompositions. HOP therefore reports two complementary views of the
    same source/target partition without defining an additional numerical
    measure:

    .. math::

       \mathrm{HOP}(f)
       = \left(\mathrm{PIRD}(f),\,\mathrm{PDGC}(f)\right).

    Parameters
    ----------
    system
        Canonical VAR or innovations state-space model, batched or unbatched.
    sources
        Exactly two or three pairwise-disjoint source groups.
    target
        Target observation group, disjoint from all source groups.
    frequencies
        One-dimensional frequency grid in physical units.
    sampling_frequency
        Positive sampling frequency associated with ``frequencies``.
    base
        Common logarithm base for information rates and Granger causality.

    Returns
    -------
    SpectralHOPResult
        Matched spectral PIRD and PDGC results sharing source/target grouping,
        source-subset ordering, PID lattice, frequency grid, dtype, device, and
        leading Torch batch dimensions.
    """
    innovations = _as_innovations(system)
    pird = spectral_partial_information_rate_decomposition(
        innovations,
        sources,
        target,
        frequencies,
        sampling_frequency=sampling_frequency,
        base=base,
    )
    pdgc = spectral_partial_granger_causality_decomposition(
        innovations,
        sources,
        target,
        frequencies,
        sampling_frequency=sampling_frequency,
        base=base,
    )
    _validate_matched_metadata(pird, pdgc)
    return SpectralHOPResult(
        frequencies=pird.frequencies,
        sources=pird.sources,
        target=pird.target,
        pird=pird,
        pdgc=pdgc,
    )


def hop_analysis(
    system: VARSystem | InnovationsStateSpace,
    sources: Sequence[int | Sequence[int]],
    target: int | Sequence[int],
    frequencies: torch.Tensor,
    *,
    sampling_frequency: float = 1.0,
    base: float = math.e,
    half_open: bool = False,
) -> HOPResult:
    r"""Compute matched whole-band PIRD and PDGC analyses.

    ``half_open=True`` applies the exact Faes/HOP half-open uniform-grid
    convention to both component decompositions. Otherwise both use the shared
    endpoint-inclusive trapezoidal integration convention.

    Parameters
    ----------
    system, sources, target, frequencies, sampling_frequency, base
        See :func:`spectral_hop_analysis`.
    half_open
        Apply the common Faes/HOP half-open integration convention to PIRD and
        PDGC simultaneously.

    Returns
    -------
    HOPResult
        Matched integrated PIRD and PDGC results.
    """
    innovations = _as_innovations(system)
    pird = partial_information_rate_decomposition(
        innovations,
        sources,
        target,
        frequencies,
        sampling_frequency=sampling_frequency,
        base=base,
        half_open=half_open,
    )
    pdgc = partial_granger_causality_decomposition(
        innovations,
        sources,
        target,
        frequencies,
        sampling_frequency=sampling_frequency,
        base=base,
        half_open=half_open,
    )
    _validate_matched_metadata(pird, pdgc)
    return HOPResult(sources=pird.sources, target=pird.target, pird=pird, pdgc=pdgc)
