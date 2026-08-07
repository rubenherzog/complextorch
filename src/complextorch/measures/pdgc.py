r"""Partial decomposition of Granger causality for Gaussian processes.

PDGC decomposes the directed Granger-causal influence from a collection of
source processes to one target process into PID-style atoms. Following the HOP
implementation of Faes et al., one first computes the unconditional spectral
Granger causality from every non-empty source subset to the target. For each
redundancy-lattice antichain, the redundant GC function is the pointwise
minimum of those subset GC spectra. Möbius inversion then yields atomic GC
spectra, which can be coarse-grained into unique, redundant, and synergistic
causal contributions.

The implementation reuses ComplexTorch's exact state-space spectral MVGC,
generic PID lattice, PIRD coarse-graining, and common spectral integrator. It
does not introduce a second state-space reduction, DARE solver, GC kernel, or
Möbius implementation. All leading Torch batch dimensions are preserved;
Python loops enumerate only source subsets and antichains, never batch items.

References
----------
- Geweke, J. (1982). Measurement of linear dependence and feedback between
  multiple time series. *Journal of the American Statistical Association*,
  77, 304-313.
- Barnett, L. and Seth, A. K. (2015). Granger causality for state-space models.
  *Physical Review E*, 91, 040101.
- Faes, L. et al. (2022). A new framework for the time- and frequency-domain
  assessment of high-order interactions in networks of random processes.
  *IEEE Transactions on Signal Processing*, 70, 5766-5777.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ..control import InnovationsStateSpace
from ..representations import VARSystem
from ..spectra import integrate_spectral_rate
from ._pid_lattice import Antichain, Subset, pid_lattice, pid_mobius_inversion
from .mvgc import state_space_spectral_mvgc
from .oir import Group
from .pird import (
    _coarse_grain_atoms,
    _indices_for_subset,
    _normalise_pird_groups,
    _source_subsets,
)
from .rates import _validate_log_base


@dataclass(frozen=True)
class SpectralPDGCResult:
    """Frequency-resolved partial decomposition of Granger causality.

    Attributes
    ----------
    frequencies
        One-dimensional physical-frequency grid supplied by the caller.
    sources
        Normalized source observation groups.
    target
        Normalized target observation group.
    source_subsets
        Non-empty subsets of source-group positions indexing ``subset_gc``.
    antichains
        PID redundancy-lattice antichains indexing ``redundancy`` and ``atoms``.
    subset_gc
        Spectral GC from each source subset to the target, shape
        ``(..., n_subsets, n_frequency)``.
    redundancy
        Redundant GC functions, shape ``(..., n_antichains, n_frequency)``.
    atoms
        Möbius-inverted atomic GC spectra with the same shape as ``redundancy``.
    unique
        Coarse-grained unique causal spectra, one per source, shape
        ``(..., n_sources, n_frequency)``.
    redundant, synergistic, delta
        Coarse-grained causal spectra, where ``delta = redundant - synergistic``.
    """

    frequencies: torch.Tensor
    sources: tuple[Group, ...]
    target: Group
    source_subsets: tuple[Subset, ...]
    antichains: tuple[Antichain, ...]
    subset_gc: torch.Tensor
    redundancy: torch.Tensor
    atoms: torch.Tensor
    unique: torch.Tensor
    redundant: torch.Tensor
    synergistic: torch.Tensor
    delta: torch.Tensor


@dataclass(frozen=True)
class PDGCResult:
    """Integrated partial decomposition of Granger causality.

    Tensor shapes equal those of :class:`SpectralPDGCResult` with the final
    frequency axis removed. ``unique`` retains its source axis.
    """

    sources: tuple[Group, ...]
    target: Group
    source_subsets: tuple[Subset, ...]
    antichains: tuple[Antichain, ...]
    subset_gc: torch.Tensor
    redundancy: torch.Tensor
    atoms: torch.Tensor
    unique: torch.Tensor
    redundant: torch.Tensor
    synergistic: torch.Tensor
    delta: torch.Tensor


def _as_innovations_system(system: VARSystem | InnovationsStateSpace) -> InnovationsStateSpace:
    """Return an innovations model while preserving batch dimensions."""
    from ..control import var_to_innovations_state_space

    if isinstance(system, VARSystem):
        return var_to_innovations_state_space(system)
    if isinstance(system, InnovationsStateSpace):
        return system
    raise TypeError("system must be a VARSystem or InnovationsStateSpace")


def spectral_partial_granger_causality_decomposition(
    system: VARSystem | InnovationsStateSpace,
    sources: Sequence[int | Sequence[int]],
    target: int | Sequence[int],
    frequencies: torch.Tensor,
    *,
    sampling_frequency: float = 1.0,
    base: float = math.e,
) -> SpectralPDGCResult:
    r"""Compute frequency-resolved partial Granger-causality decomposition.

    For every non-empty source subset :math:`A`, let
    :math:`f_{A\to Y}(\nu)` denote its unconditional state-space spectral GC to
    the target after exact marginalization to ``(A, Y)``. For an antichain
    :math:`\alpha`, HOP defines the redundant GC function

    .. math::

       F_{\cap}(\alpha;\nu)
       = \min_{A\in\alpha} f_{A\to Y}(\nu).

    Atomic causal spectra are obtained by Möbius inversion of these redundancy
    functions on the Williams--Beer lattice.

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
        Positive sampling frequency. Internally frequencies are normalized to
        cycles/sample for the shared state-space GC kernel.
    base
        Logarithm base for Granger causality. Defaults to natural units.

    Returns
    -------
    SpectralPDGCResult
        Subset GC spectra, redundancy functions, PID atoms, and coarse-grained
        unique/redundant/synergistic causal spectra.

    Notes
    -----
    ``conditional=()`` is passed explicitly to the shared GC kernel. This is
    essential: HOP PDGC first marginalizes to each source subset plus target and
    computes unconditional GC, rather than conditioning on omitted channels.
    """
    if sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be positive")
    _validate_log_base(base)
    innovations = _as_innovations_system(system)
    source_groups, target_group = _normalise_pird_groups(
        innovations, sources, target
    )
    frequency = torch.as_tensor(
        frequencies,
        dtype=innovations.transition.dtype,
        device=innovations.transition.device,
    )
    if frequency.ndim != 1 or frequency.numel() == 0:
        raise ValueError("frequencies must be a non-empty one-dimensional tensor")
    if not bool(torch.isfinite(frequency).all().item()):
        raise ValueError("frequencies must contain only finite values")
    if bool(torch.any((frequency < 0) | (frequency > sampling_frequency / 2)).item()):
        raise ValueError("frequencies must lie in [0, sampling_frequency/2]")

    normalized_frequency = frequency / float(sampling_frequency)
    lattice = pid_lattice(len(source_groups), device=innovations.transition.device)
    subsets = _source_subsets(len(source_groups))
    subset_rates: dict[Subset, torch.Tensor] = {}
    for subset in subsets:
        subset_rates[subset] = state_space_spectral_mvgc(
            innovations,
            source=_indices_for_subset(source_groups, subset),
            target=target_group,
            frequencies=normalized_frequency,
            conditional=(),
            base=base,
        )

    subset_gc = torch.stack([subset_rates[subset] for subset in subsets], dim=-2)
    redundancy_frequency_last = torch.stack(
        [
            torch.stack([subset_rates[subset] for subset in antichain], dim=-1)
            .amin(dim=-1)
            for antichain in lattice.antichains
        ],
        dim=-1,
    )
    atoms_frequency_last = pid_mobius_inversion(redundancy_frequency_last, lattice)
    redundancy = redundancy_frequency_last.movedim(-1, -2)
    atoms = atoms_frequency_last.movedim(-1, -2)
    unique, redundant, synergistic, delta = _coarse_grain_atoms(atoms, lattice)
    return SpectralPDGCResult(
        frequencies=frequency,
        sources=source_groups,
        target=target_group,
        source_subsets=subsets,
        antichains=lattice.antichains,
        subset_gc=subset_gc,
        redundancy=redundancy,
        atoms=atoms,
        unique=unique,
        redundant=redundant,
        synergistic=synergistic,
        delta=delta,
    )


def partial_granger_causality_decomposition(
    system: VARSystem | InnovationsStateSpace,
    sources: Sequence[int | Sequence[int]],
    target: int | Sequence[int],
    frequencies: torch.Tensor,
    *,
    sampling_frequency: float = 1.0,
    base: float = math.e,
    half_open: bool = False,
) -> PDGCResult:
    r"""Compute integrated partial Granger-causality decomposition.

    The redundancy principle is frequency-resolved, so the temporal PDGC atoms
    are integrals of spectral atoms. With ``half_open=True`` and
    ``f_k = k Fs/(2N)``, integration reduces to the arithmetic mean used by the
    original HOP ``hop_pdgc`` implementation.

    Parameters
    ----------
    system, sources, target, frequencies, sampling_frequency, base
        See :func:`spectral_partial_granger_causality_decomposition`.
    half_open
        Use the exact HOP/Faes half-open uniform-grid integration convention.
        Otherwise use endpoint-inclusive trapezoidal integration.

    Returns
    -------
    PDGCResult
        Integrated subset GCs, redundancy functions, atomic causal terms, and
        coarse-grained unique/redundant/synergistic causal contributions.
    """
    spectral = spectral_partial_granger_causality_decomposition(
        system,
        sources,
        target,
        frequencies,
        sampling_frequency=sampling_frequency,
        base=base,
    )

    def integrate(value: torch.Tensor) -> torch.Tensor:
        """Integrate a PDGC result tensor along its final frequency axis."""
        return integrate_spectral_rate(
            value,
            spectral.frequencies,
            sampling_frequency=sampling_frequency,
            half_open=half_open,
        )

    return PDGCResult(
        sources=spectral.sources,
        target=spectral.target,
        source_subsets=spectral.source_subsets,
        antichains=spectral.antichains,
        subset_gc=integrate(spectral.subset_gc),
        redundancy=integrate(spectral.redundancy),
        atoms=integrate(spectral.atoms),
        unique=integrate(spectral.unique),
        redundant=integrate(spectral.redundant),
        synergistic=integrate(spectral.synergistic),
        delta=integrate(spectral.delta),
    )
