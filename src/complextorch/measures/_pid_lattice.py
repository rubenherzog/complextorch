"""Generic redundancy-lattice primitives for PID-style decompositions.

The lattice contains antichains of non-empty source subsets. Its order is the
Williams--Beer redundancy order: ``alpha <= beta`` iff every element of beta
contains at least one element of alpha. The implementation is measure-agnostic
and is shared by future PIRD and PDGC code.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import torch

Subset = frozenset[int]
Antichain = tuple[Subset, ...]


def _subset_mask(subset: Subset) -> int:
    """Encode a zero-based source subset as an integer bit mask."""
    mask = 0
    for index in subset:
        mask |= 1 << index
    return mask


def _antichain_key(antichain: Antichain) -> tuple[int, tuple[int, ...]]:
    """Return a deterministic sort key for PID antichains."""
    return (len(antichain), tuple(sorted(_subset_mask(item) for item in antichain)))


def redundancy_leq(alpha: Antichain, beta: Antichain) -> bool:
    """Return whether ``alpha <= beta`` in the PID redundancy lattice."""
    return all(any(left.issubset(right) for left in alpha) for right in beta)


def _all_antichains(n_sources: int) -> list[Antichain]:
    """Enumerate all antichains of non-empty subsets for ``n_sources``."""
    if not 1 <= n_sources <= 4:
        raise ValueError("n_sources must be between 1 and 4")
    subsets = [
        frozenset(index for index in range(n_sources) if mask & (1 << index))
        for mask in range(1, 1 << n_sources)
    ]
    antichains: list[Antichain] = []
    for size in range(1, len(subsets) + 1):
        for candidate in combinations(subsets, size):
            if all(
                not left.issubset(right) and not right.issubset(left)
                for left, right in combinations(candidate, 2)
            ):
                antichains.append(
                    tuple(sorted(candidate, key=lambda item: (_subset_mask(item), len(item))))
                )
    return antichains


def _topological_antichains(n_sources: int) -> tuple[Antichain, ...]:
    """Topologically order PID antichains under the redundancy relation."""
    nodes = _all_antichains(n_sources)
    remaining = set(nodes)
    ordered: list[Antichain] = []
    while remaining:
        minimal = [
            node
            for node in remaining
            if not any(
                other != node
                and redundancy_leq(other, node)
                and not redundancy_leq(node, other)
                for other in remaining
            )
        ]
        if not minimal:
            raise RuntimeError("PID redundancy order contains a cycle")
        for node in sorted(minimal, key=_antichain_key):
            ordered.append(node)
            remaining.remove(node)
    return tuple(ordered)


@dataclass(frozen=True)
class PIDLattice:
    """Finite PID redundancy lattice with deterministic topological ordering."""

    n_sources: int
    antichains: tuple[Antichain, ...]
    zeta: torch.Tensor
    mobius: torch.Tensor

    def index(self, antichain) -> int:
        """Return the index of an antichain expressed as iterable source sets."""
        canonical = tuple(
            sorted(
                (frozenset(int(item) for item in subset) for subset in antichain),
                key=lambda item: (_subset_mask(item), len(item)),
            )
        )
        return self.antichains.index(canonical)


def pid_lattice(n_sources: int, *, device=None) -> PIDLattice:
    """Construct antichains, zeta matrix and Möbius matrix for ``n_sources``.

    Source labels are zero-based. ``zeta[i,j]=1`` when antichain ``j`` is below
    antichain ``i``. Therefore ``redundancy = zeta @ atoms`` and
    ``atoms = mobius @ redundancy``.
    """
    antichains = _topological_antichains(n_sources)
    count = len(antichains)
    zeta = torch.zeros((count, count), dtype=torch.int64, device=device)
    for i, upper in enumerate(antichains):
        for j, lower in enumerate(antichains):
            if redundancy_leq(lower, upper):
                zeta[i, j] = 1

    # The topological order makes zeta unit lower triangular. Compute the
    # integer Möbius matrix by exact forward substitution, not matrix inverse.
    mobius = torch.zeros_like(zeta)
    for column in range(count):
        solution = torch.zeros(count, dtype=torch.int64, device=device)
        for row in range(count):
            rhs = 1 if row == column else 0
            if row:
                rhs -= int((zeta[row, :row] * solution[:row]).sum().item())
            solution[row] = rhs
        mobius[:, column] = solution
    return PIDLattice(n_sources, antichains, zeta, mobius)


def pid_mobius_inversion(
    redundancy: torch.Tensor,
    lattice: PIDLattice,
) -> torch.Tensor:
    """Convert redundancy values to PID atoms along the last tensor axis."""
    value = torch.as_tensor(redundancy)
    if value.shape[-1] != len(lattice.antichains):
        raise ValueError("last redundancy dimension must match the PID lattice")
    transform = lattice.mobius.to(dtype=value.dtype, device=value.device)
    return value @ transform.transpose(-1, -2)


def pid_redundancy_from_atoms(
    atoms: torch.Tensor,
    lattice: PIDLattice,
) -> torch.Tensor:
    """Reconstruct redundancy-lattice values from atoms along the last axis."""
    value = torch.as_tensor(atoms)
    if value.shape[-1] != len(lattice.antichains):
        raise ValueError("last atom dimension must match the PID lattice")
    transform = lattice.zeta.to(dtype=value.dtype, device=value.device)
    return value @ transform.transpose(-1, -2)
