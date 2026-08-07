"""Reusable analysis primitives for the SSDI macro-dimension study.

This module is deliberately outside :mod:`complextorch`: it is analysis and
validation infrastructure, not a new public package API.  It preserves the
ComplexTorch row-projection convention throughout:

``projection.shape == (runs, macro_dimension, microscopic_dimension)``.

The heavy experiment requested for this branch (100 random restarts, up to
10,000 gradient iterations, every macro dimension) will use these primitives
after the lightweight validation layer passes.

The Grassmann distance functions reproduce the semantics of SSDI/ComplexBox
``subspacea.m`` + ``gmetric.m``/``gmetrics.m``.  ComplexBox calls the resulting
matrix a metric/distance matrix even when downstream analyses refer to it
informally as a similarity matrix.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from complextorch import DDOptimizationResult, InnovationsStateSpace, optimise_dynamical_dependence
from scripts.validate_ssdi import innovations_system_from_mask, random_initial_projections


def fixed_modular_8_mask(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Return the predefined 8-node, 2+6 modular directed connectivity mask.

    Matrix orientation follows the state equation ``x[t+1] = A x[t]``:
    ``mask[target, source] != 0`` denotes the directed edge ``source -> target``.

    Nodes 0--1 form the 2-node module and nodes 2--7 form the 6-node module.
    Both modules are internally fully connected (including self dynamics).
    There is exactly one inter-module edge, from node 1 in the 2-node module to
    node 2 in the 6-node module, and no edge in the reverse direction.  This is
    the fixed starting topology for the collaborator-requested SSDI study.
    """
    mask = torch.zeros((8, 8), dtype=dtype)
    mask[:2, :2] = 1.0
    mask[2:, 2:] = 1.0
    mask[2, 1] = 1.0  # source node 1 -> target node 2
    return mask


def fixed_modular_8_system(
    *,
    seed: int = 20260807,
    target_radius: float = 0.72,
    gain_scale: float = 0.16,
    dtype: torch.dtype = torch.float64,
) -> InnovationsStateSpace:
    """Build the stable innovations-form SSM for :func:`fixed_modular_8_mask`.

    The connectivity is fixed while nonzero transition weights are generated
    from a pinned random seed and rescaled to ``target_radius``.  Observation,
    innovation covariance, and gain conventions are the same as the other
    deterministic SSDI validation systems in ``scripts.validate_ssdi``.
    """
    return innovations_system_from_mask(
        fixed_modular_8_mask(dtype),
        seed=seed,
        target_radius=target_radius,
        gain_scale=gain_scale,
    )


def macro_dimensions(system: InnovationsStateSpace) -> tuple[int, ...]:
    """Return every nontrivial macro dimension ``1, ..., n_observations - 1``."""
    n = int(system.observation.shape[-2])
    if n < 2:
        raise ValueError("the SSDI macro sweep requires at least two observations")
    return tuple(range(1, n))


def run_macro_dimension_sweep(
    system: InnovationsStateSpace,
    *,
    restarts: int,
    max_iterations: int,
    optimizer: str = "complexbox",
    objective: str = "proxy",
    seed: int = 20260807,
    lags: int | None = None,
    frequencies: torch.Tensor | None = None,
    history: bool = True,
    optimizer_options: Mapping[str, Any] | None = None,
) -> dict[int, DDOptimizationResult]:
    """Optimize DD independently for every nontrivial macro dimension.

    Parameters
    ----------
    system
        One microscopic innovations-form model.  The heavy study starts from
        :func:`fixed_modular_8_system`, but this helper is dimension agnostic.
    restarts
        Number of independent random Grassmann initializations for *each* macro
        dimension.  In the requested heavy experiment this will be 100.
    max_iterations
        Per-restart optimizer iteration limit.  In the requested heavy
        experiment this will be 10,000.
    optimizer, objective
        Arguments forwarded to :func:`complextorch.optimise_dynamical_dependence`.
    seed
        Base seed.  Macro dimension ``m`` uses ``seed + m`` so its restart set
        is deterministic but independent of the other dimensions.
    lags, frequencies
        Objective-specific arguments.  ``lags`` defaults to the latent state
        dimension for the proxy objective, matching the existing SSDI contract.
    history
        Retain optimizer trajectories.  This must be ``True`` for the later
        DD-versus-iteration figure.
    optimizer_options
        Backend-specific options forwarded without changing the public API.

    Returns
    -------
    dict[int, DDOptimizationResult]
        One sorted optimizer result per macro dimension.  Every projection uses
        the repository-native orientation ``(restarts, m, n)``; no MATLAB-style
        transposition is introduced or stored.
    """
    restarts = int(restarts)
    max_iterations = int(max_iterations)
    if restarts < 1:
        raise ValueError("restarts must be positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")

    n = int(system.observation.shape[-2])
    dtype = system.observation.dtype
    device = system.observation.device
    results: dict[int, DDOptimizationResult] = {}
    for m in macro_dimensions(system):
        initial = random_initial_projections(
            restarts,
            m,
            n,
            seed=seed + m,
            dtype=dtype,
            device=device,
        )
        results[m] = optimise_dynamical_dependence(
            system,
            initial,
            objective=objective,
            optimizer=optimizer,
            lags=lags,
            frequencies=frequencies,
            max_iterations=max_iterations,
            history=history,
            optimizer_options=optimizer_options,
        )
    return results


def _row_orthonormal_basis(projection: torch.Tensor) -> torch.Tensor:
    """Return a numerically orthonormal row basis without changing its span."""
    matrix = torch.as_tensor(projection)
    if matrix.ndim < 2:
        raise ValueError("projection must have at least two dimensions")
    if matrix.shape[-2] > matrix.shape[-1]:
        raise ValueError("macro dimension cannot exceed microscopic dimension")
    q, _ = torch.linalg.qr(matrix.transpose(-1, -2), mode="reduced")
    return q.transpose(-1, -2)


def principal_angles_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return principal angles between equal-dimensional row subspaces.

    The calculation is the row-oriented equivalent of ComplexBox ``subspacea``:
    singular values of ``Q_a Q_b.T`` are cosines of principal angles.  Values
    are clamped to ``[0, 1]`` before ``acos`` for floating-point robustness.
    """
    left = torch.as_tensor(a)
    right = torch.as_tensor(b, dtype=left.dtype, device=left.device)
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise ValueError("a and b must have the same two-dimensional shape")
    qa = _row_orthonormal_basis(left)
    qb = _row_orthonormal_basis(right)
    cosines = torch.linalg.svdvals(qa @ qb.transpose(-1, -2)).clamp(0.0, 1.0)
    return torch.acos(cosines)


def grassmann_distance(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    max_angle: bool = True,
) -> torch.Tensor:
    """Return the SSDI/ComplexBox normalized Grassmann distance in ``[0, 1]``.

    This matches ``gmetric``: largest principal angle divided by ``pi/2`` when
    ``max_angle=True`` (the reference default), otherwise RMS principal angle
    divided by ``pi/2``.
    """
    angles = principal_angles_rows(a, b)
    if max_angle:
        value = torch.max(angles)
    else:
        value = torch.sqrt(torch.mean(angles.square()))
    return value / (math.pi / 2.0)


def grassmann_distance_matrix(
    projections: torch.Tensor,
    *,
    max_angle: bool = True,
) -> torch.Tensor:
    """Return the pairwise ``gmetrics`` distance matrix for optimizer endpoints.

    Parameters
    ----------
    projections
        ComplexTorch row projections with shape ``(runs, m, n)``.
    max_angle
        ``True`` reproduces the default ComplexBox/MATLAB ``gmetrics`` metric.

    Returns
    -------
    torch.Tensor
        Symmetric ``(runs, runs)`` matrix with exactly zero diagonal.
    """
    matrices = torch.as_tensor(projections)
    if matrices.ndim != 3:
        raise ValueError("projections must have shape (runs, m, n)")
    runs = matrices.shape[0]
    if runs < 1:
        raise ValueError("projections must contain at least one run")
    out = torch.zeros((runs, runs), dtype=matrices.dtype, device=matrices.device)
    for i in range(runs):
        for j in range(i + 1, runs):
            value = grassmann_distance(matrices[i], matrices[j], max_angle=max_angle)
            out[i, j] = value
            out[j, i] = value
    return out


def micro_macro_loadings(projection: torch.Tensor) -> torch.Tensor:
    """Return basis-invariant squared micro-unit loadings onto a macro subspace.

    For a row-orthonormal ComplexTorch projection ``M`` with shape ``(...,m,n)``,
    the loading of microscopic coordinate ``i`` is

    ``h_i = sum_j M[..., j, i]**2``.

    This is exactly ComplexBox/SSDI ``habeta(L)`` under ``L = M.T``.  It is
    invariant to rotations of the basis *within* the same macro subspace, lies
    in ``[0,1]``, and sums to ``m``.  Those properties make it the appropriate
    micro-to-macro loading statistic rather than raw, basis-dependent entries of
    ``M``.
    """
    matrix = _row_orthonormal_basis(torch.as_tensor(projection))
    return torch.sum(matrix.square(), dim=-2)


def coordinate_axis_distances(projection: torch.Tensor) -> torch.Tensor:
    """Return SSDI ``gmetrics1`` distances from each micro axis to the subspace.

    The output is basis invariant and normalized to ``[0,1]``.  Zero means the
    microscopic coordinate axis lies in the macro subspace; one means it is
    orthogonal.  This is a monotone transformation of
    :func:`micro_macro_loadings` and is retained because it is the exact
    ``gmetrics1.m`` quantity requested for external parity.
    """
    loadings = micro_macro_loadings(projection).clamp(0.0, 1.0)
    return torch.acos(torch.sqrt(loadings)) / (math.pi / 2.0)


__all__ = [
    "coordinate_axis_distances",
    "fixed_modular_8_mask",
    "fixed_modular_8_system",
    "grassmann_distance",
    "grassmann_distance_matrix",
    "macro_dimensions",
    "micro_macro_loadings",
    "principal_angles_rows",
    "run_macro_dimension_sweep",
]
