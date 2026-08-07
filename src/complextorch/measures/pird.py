r"""Partial information rate decomposition for Gaussian random processes.

PIRD decomposes the mutual-information rate between a target process and a
collection of source processes into PID atoms.  Following Faes and the HOP
implementation, the redundancy function is the frequency-wise minimum of the
source-subset mutual-information-rate spectra.  Möbius inversion on the
Williams--Beer redundancy lattice then yields the spectral atoms, which are
integrated to obtain temporal partial information rates.

This module intentionally reuses the shared Gaussian MIR, spectral integration,
and generic PID-lattice primitives.  It does not implement an alternative
state-space reduction, DARE solver, spectral density, or Möbius algorithm.
All leading batch dimensions are preserved natively by Torch; Python loops only
enumerate source subsets and lattice antichains, never batch elements.

The public implementation supports two or three source groups, matching the
PIRD/HOP decompositions for which the coarse-grained unique/redundant/
synergistic semantics are explicitly defined and independently testable.

References
----------
- Williams, P. L. and Beer, R. D. (2010). Nonnegative decomposition of
  multivariate information. arXiv:1004.2515.
- Faes, L. et al. (2022). A new framework for the time- and frequency-domain
  assessment of high-order interactions in networks of random processes.
  *IEEE Transactions on Signal Processing*, 70, 5766-5777.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

import torch

from ..control import InnovationsStateSpace
from ..spectra import integrate_spectral_rate
from ._pid_lattice import Antichain, PIDLattice, Subset, pid_lattice, pid_mobius_inversion
from .oir import Group, _flatten, _normalise_groups
from .rates import (
    _normalise_group,
    _validate_log_base,
    gaussian_mutual_information_rate,
    spectral_gaussian_mutual_information_rate,
)


@dataclass(frozen=True)
class SpectralPIRDResult:
    """Frequency-resolved partial information rate decomposition.

    Attributes
    ----------
    frequencies
        One-dimensional frequency grid.
    sources
        Normalized source observation groups.
    target
        Normalized target observation group.
    source_subsets
        Non-empty subsets of zero-based source-group positions.  The same order
        indexes ``subset_mir``.
    antichains
        PID redundancy-lattice antichains.  The same order indexes
        ``redundancy`` and ``atoms``.
    subset_mir
        Source-subset MIR spectra with shape ``(..., n_subsets, n_frequency)``.
    redundancy
        Redundancy-function spectra with shape
        ``(..., n_antichains, n_frequency)``.
    atoms
        Möbius-inverted partial information spectra with shape
        ``(..., n_antichains, n_frequency)``.
    unique
        Coarse-grained unique spectra, one per source, shape
        ``(..., n_sources, n_frequency)``.
    redundant, synergistic, delta
        Coarse-grained spectra.  ``delta = redundant - synergistic``.
    """

    frequencies: torch.Tensor
    sources: tuple[Group, ...]
    target: Group
    source_subsets: tuple[Subset, ...]
    antichains: tuple[Antichain, ...]
    subset_mir: torch.Tensor
    redundancy: torch.Tensor
    atoms: torch.Tensor
    unique: torch.Tensor
    redundant: torch.Tensor
    synergistic: torch.Tensor
    delta: torch.Tensor


@dataclass(frozen=True)
class PIRDResult:
    """Integrated partial information rate decomposition.

    Tensor shapes equal those of :class:`SpectralPIRDResult` with the final
    frequency dimension removed.  ``unique`` retains the source axis.
    """

    sources: tuple[Group, ...]
    target: Group
    source_subsets: tuple[Subset, ...]
    antichains: tuple[Antichain, ...]
    subset_mir: torch.Tensor
    redundancy: torch.Tensor
    atoms: torch.Tensor
    unique: torch.Tensor
    redundant: torch.Tensor
    synergistic: torch.Tensor
    delta: torch.Tensor


def _normalise_pird_groups(
    system: InnovationsStateSpace,
    sources: Sequence[int | Sequence[int]],
    target: int | Sequence[int],
) -> tuple[tuple[Group, ...], Group]:
    """Validate two or three disjoint source groups and one target group."""
    source_groups = _normalise_groups(system, sources)
    if len(source_groups) not in (2, 3):
        raise ValueError("PIRD currently supports exactly two or three source groups")
    n_observations = system.observation.shape[-2]
    target_group = _normalise_group(target, n_observations, name="target")
    source_indices = set(_flatten(source_groups))
    if source_indices & set(target_group):
        raise ValueError("source groups and target must be disjoint")
    return source_groups, target_group


def _source_subsets(n_sources: int) -> tuple[Subset, ...]:
    """Enumerate non-empty source-position subsets in deterministic mask order."""
    return tuple(
        frozenset(index for index in range(n_sources) if mask & (1 << index))
        for mask in range(1, 1 << n_sources)
    )


def _indices_for_subset(sources: tuple[Group, ...], subset: Subset) -> Group:
    """Concatenate observation indices for one source-position subset."""
    return _flatten(tuple(sources[index] for index in sorted(subset)))


def _coarse_grain_labels(lattice: PIDLattice) -> tuple[str, ...]:
    """Return Faes/HOP coarse-graining labels for two- or three-source atoms."""
    if lattice.n_sources == 2:
        labels = {
            ((0,),): "U0",
            ((1,),): "U1",
            ((0, 1),): "Sy",
            ((0,), (1,)): "Rd",
        }
    elif lattice.n_sources == 3:
        labels = {
            ((0,),): "U0",
            ((1,),): "U1",
            ((2,),): "U2",
            ((0, 1),): "Sy",
            ((0, 2),): "Sy",
            ((1, 2),): "Sy",
            ((0, 1, 2),): "Sy",
            ((0,), (1,)): "Rd",
            ((0,), (2,)): "Rd",
            ((0,), (1, 2)): "U0",
            ((1,), (2,)): "Rd",
            ((1,), (0, 2)): "U1",
            ((2,), (0, 1)): "U2",
            ((0, 1), (0, 2)): "Sy",
            ((0, 1), (1, 2)): "Sy",
            ((0, 2), (1, 2)): "Sy",
            ((0,), (1,), (2,)): "Rd",
            ((0, 1), (0, 2), (1, 2)): "Sy",
        }
    else:
        raise ValueError("coarse graining is defined here only for two or three sources")

    def key(antichain: Antichain) -> tuple[tuple[int, ...], ...]:
        """Convert an antichain to the exact source-position tuple key."""
        return tuple(tuple(sorted(subset)) for subset in antichain)

    try:
        return tuple(labels[key(antichain)] for antichain in lattice.antichains)
    except KeyError as exc:
        raise RuntimeError("PID lattice does not match the validated HOP coarse graining") from exc


def _coarse_grain_atoms(
    atoms: torch.Tensor,
    lattice: PIDLattice,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate atom spectra into unique, redundant, synergistic, and delta terms."""
    labels = _coarse_grain_labels(lattice)
    n_sources = lattice.n_sources
    unique_terms = []
    for source in range(n_sources):
        positions = [index for index, label in enumerate(labels) if label == f"U{source}"]
        unique_terms.append(atoms[..., positions, :].sum(dim=-2))
    unique = torch.stack(unique_terms, dim=-2)
    redundant_positions = [index for index, label in enumerate(labels) if label == "Rd"]
    synergistic_positions = [index for index, label in enumerate(labels) if label == "Sy"]
    redundant = atoms[..., redundant_positions, :].sum(dim=-2)
    synergistic = atoms[..., synergistic_positions, :].sum(dim=-2)
    return unique, redundant, synergistic, redundant - synergistic


def spectral_partial_information_rate_decomposition(
    system: InnovationsStateSpace,
    sources: Sequence[int | Sequence[int]],
    target: int | Sequence[int],
    frequencies: torch.Tensor,
    *,
    sampling_frequency: float = 1.0,
    base: float = math.e,
) -> SpectralPIRDResult:
    r"""Compute the frequency-resolved Gaussian partial information rate decomposition.

    For every non-empty source subset :math:`A`, first compute the Gaussian
    spectral mutual-information rate :math:`i_A(f)=i(X_A;Y;f)`.  For a
    redundancy-lattice antichain :math:`\alpha`, the minimum-MIR redundancy
    function is

    .. math::

       I_{\cap}(\alpha;f)=\min_{A\in\alpha} i_A(f).

    The partial information atoms are obtained by Möbius inversion of these
    redundancy functions on the Williams--Beer lattice.

    Parameters
    ----------
    system
        Exact innovations-form process, batched or unbatched.
    sources
        Exactly two or three disjoint source groups.  Each group is an
        observation index or a sequence of observation indices.
    target
        Target observation group, disjoint from every source group.
    frequencies
        One-dimensional frequency grid in cycles per unit time.
    sampling_frequency
        Positive sampling frequency associated with ``frequencies``.
    base
        Logarithm base.  Defaults to natural units.

    Returns
    -------
    SpectralPIRDResult
        Source-subset MIR spectra, lattice redundancy functions, PID atoms, and
        Faes/HOP coarse-grained unique/redundant/synergistic spectra.

    Notes
    -----
    The implementation is batch-native.  Loops enumerate at most seven source
    subsets and 18 antichains; no loop iterates over batch elements.
    """
    _validate_log_base(base)
    source_groups, target_group = _normalise_pird_groups(system, sources, target)
    lattice = pid_lattice(len(source_groups), device=system.transition.device)
    subsets = _source_subsets(len(source_groups))

    subset_rates: dict[Subset, torch.Tensor] = {}
    for subset in subsets:
        subset_rates[subset] = spectral_gaussian_mutual_information_rate(
            system,
            _indices_for_subset(source_groups, subset),
            target_group,
            frequencies,
            sampling_frequency=sampling_frequency,
            base=base,
        )

    subset_mir = torch.stack([subset_rates[subset] for subset in subsets], dim=-2)
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
    frequency = torch.as_tensor(
        frequencies,
        dtype=system.transition.dtype,
        device=system.transition.device,
    )
    return SpectralPIRDResult(
        frequencies=frequency,
        sources=source_groups,
        target=target_group,
        source_subsets=subsets,
        antichains=lattice.antichains,
        subset_mir=subset_mir,
        redundancy=redundancy,
        atoms=atoms,
        unique=unique,
        redundant=redundant,
        synergistic=synergistic,
        delta=delta,
    )


def partial_information_rate_decomposition(
    system: InnovationsStateSpace,
    sources: Sequence[int | Sequence[int]],
    target: int | Sequence[int],
    frequencies: torch.Tensor,
    *,
    sampling_frequency: float = 1.0,
    base: float = math.e,
    half_open: bool = False,
) -> PIRDResult:
    r"""Compute integrated Gaussian partial information rates.

    PIRD's minimum redundancy is defined frequency by frequency, so temporal
    atoms are the integrals of the spectral atoms rather than a decomposition
    obtained by applying a minimum to already integrated MIRs.

    Parameters
    ----------
    system, sources, target, frequencies, sampling_frequency, base
        See :func:`spectral_partial_information_rate_decomposition`.
    half_open
        If ``True``, use the exact Faes/HOP half-open grid convention and
        arithmetic-mean integration implemented by
        :func:`complextorch.integrate_spectral_rate`.  Otherwise use trapezoidal
        integration on an endpoint-inclusive grid.

    Returns
    -------
    PIRDResult
        Integrated subset MIRs, redundancy functions, PID atoms, and
        coarse-grained unique/redundant/synergistic rates.
    """
    spectral = spectral_partial_information_rate_decomposition(
        system,
        sources,
        target,
        frequencies,
        sampling_frequency=sampling_frequency,
        base=base,
    )

    def integrate(value: torch.Tensor) -> torch.Tensor:
        """Integrate one result tensor whose final axis is frequency."""
        return integrate_spectral_rate(
            value,
            spectral.frequencies,
            sampling_frequency=sampling_frequency,
            half_open=half_open,
        )

    return PIRDResult(
        sources=spectral.sources,
        target=spectral.target,
        source_subsets=spectral.source_subsets,
        antichains=spectral.antichains,
        subset_mir=integrate(spectral.subset_mir),
        redundancy=integrate(spectral.redundancy),
        atoms=integrate(spectral.atoms),
        unique=integrate(spectral.unique),
        redundant=integrate(spectral.redundant),
        synergistic=integrate(spectral.synergistic),
        delta=integrate(spectral.delta),
    )


def direct_subset_mutual_information_rates(
    system: InnovationsStateSpace,
    sources: Sequence[int | Sequence[int]],
    target: int | Sequence[int],
    *,
    base: float = math.e,
) -> tuple[tuple[Subset, ...], torch.Tensor]:
    """Return exact temporal source-subset MIRs for validation and diagnostics.

    This helper does not define PIRD atoms.  It provides the exact temporal MIR
    values against which integrated spectral subset MIRs and reconstructed PID
    lattice nodes can be checked.
    """
    source_groups, target_group = _normalise_pird_groups(system, sources, target)
    subsets = _source_subsets(len(source_groups))
    values = torch.stack(
        [
            gaussian_mutual_information_rate(
                system,
                _indices_for_subset(source_groups, subset),
                target_group,
                base=base,
            )
            for subset in subsets
        ],
        dim=-1,
    )
    return subsets, values
