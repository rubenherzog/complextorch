"""Staged SSDI dynamical-independence optimisation workflow.

This module implements the *algorithmic workflow* used by the Barnett--Seth
MATLAB SSDI toolbox and the audited ComplexBox port while keeping ComplexTorch's
row-projection convention and native Torch numerical kernels:

1. whiten innovations / transform projection coordinates;
2. run many proxy-DD (``ddx``) pre-optimisations;
3. compute pairwise Grassmann distances and apply SSDI ``Lcluster``;
4. refine one representative per cluster with spectral DD (``dds``);
5. inverse-transform the final projections to physical observation coordinates.

For a :class:`~complextorch.representations.VARSystem`, the proxy stage uses the
transformed VAR coefficient sequence directly, matching SSDI ``cak2ddx``.  The
spectral stage uses the exactly equivalent innovations state-space transfer
function through the existing ComplexTorch backend.

References
----------
- Barnett, L. and Seth, A. K. (2023). Dynamical independence: Discovering
  emergent macroscopic processes in complex dynamical systems. Physical Review
  E 108, 014304.
- ``lcbarnett/ssdi`` commit ``b38ce65f9df18916da216848560c1789e456c04f``:
  ``transform_var.m``, ``transform_subspace.m``, ``itransform_subspace.m``,
  ``Lcluster.m``, ``opt_gd_ddx_mruns.m``, and ``opt_gd_dds_mruns.m``.
- ``bmilinkovic/complexbox`` commit
  ``87b5e2cd9bba22ddd978bade6f614da7d6190db2``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, Mapping

import torch

from .dd_optimization import (
    DDGradientSearchResult,
    Model,
    _optimise,
    _projection_to_whitened,
    _restore_result_coordinates,
    _single_innovations,
    optimise_dynamical_dependence_proxy,
    optimise_dynamical_dependence_spectral,
    orthonormalise_projection,
    proxy_dynamical_dependence,
    proxy_dynamical_dependence_gradient,
)
from .dd_riemannian import (
    DDRiemannianSearchResult,
    _restore_physical_coordinates,
    _riemannian_armijo,
    optimise_dynamical_dependence_proxy_riemannian,
    optimise_dynamical_dependence_spectral_riemannian,
)
from .representations import VARSystem

DDOptimizer = Literal["complexbox", "riemannian_armijo"]
RawSearchResult = DDGradientSearchResult | DDRiemannianSearchResult


@dataclass(frozen=True)
class SSDIRawResult:
    """Raw backend results and clustering metadata for the staged workflow."""

    preoptimization: RawSearchResult
    spectral: RawSearchResult
    cluster_indices: torch.Tensor
    cluster_sizes: torch.Tensor
    cluster_distances: torch.Tensor


def _single_var_proxy_problem(
    system: VARSystem,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    r"""Return whitened VAR coefficients for SSDI ``cak2ddx`` pre-optimisation.

    For innovation covariance :math:`V=BB^\top`, MATLAB ``transform_var`` uses

    .. math::

       A_k^{(w)} = B^{-1} A_k B.

    ``factor`` is returned because the matching row-projection transform is
    ``M_w = orth(M B)`` and the inverse transform is ``M = orth(M_w B^{-1})``.
    """
    coefficients = torch.as_tensor(system.coefficients)
    covariance = torch.as_tensor(system.innovation_covariance)
    if coefficients.ndim != 4 or coefficients.shape[0] != 1:
        raise ValueError(
            "staged SSDI currently accepts one microscopic VAR system"
        )
    if covariance.ndim == 3:
        if covariance.shape[0] != 1:
            raise ValueError(
                "staged SSDI currently accepts one VAR innovation covariance"
            )
        covariance = covariance[0]
    if covariance.ndim != 2:
        raise ValueError("VAR innovation covariance must be two-dimensional")
    coefficients = coefficients[0]
    identity = torch.eye(
        covariance.shape[-1], dtype=covariance.dtype, device=covariance.device
    )
    identity_coordinates = bool(
        torch.allclose(covariance, identity, rtol=1e-7, atol=1e-9)
    )
    if identity_coordinates:
        return coefficients, identity, True
    try:
        factor = torch.linalg.cholesky(covariance)
    except RuntimeError as exc:
        raise ValueError("innovation covariance must be positive definite") from exc
    transformed = torch.linalg.solve_triangular(
        factor.expand(coefficients.shape[0], -1, -1),
        coefficients @ factor,
        upper=False,
    )
    return transformed, factor, False


def _optimise_var_proxy_complexbox(
    system: VARSystem,
    initial_projection: torch.Tensor,
    **options: Any,
) -> DDGradientSearchResult:
    """ComplexBox-compatible proxy search using the transformed VAR sequence."""
    sequence, factor, identity_coordinates = _single_var_proxy_problem(system)
    initial = _projection_to_whitened(
        initial_projection,
        factor,
        identity_coordinates=identity_coordinates,
    ).to(dtype=sequence.dtype, device=sequence.device)
    result = _optimise(
        initial,
        objective=lambda matrix: proxy_dynamical_dependence(matrix, sequence),
        gradient=lambda matrix: proxy_dynamical_dependence_gradient(
            matrix, sequence
        ),
        spectral=False,
        **options,
    )
    return _restore_result_coordinates(
        result,
        factor,
        identity_coordinates=identity_coordinates,
    )


def _optimise_var_proxy_riemannian(
    system: VARSystem,
    initial_projection: torch.Tensor,
    **options: Any,
) -> DDRiemannianSearchResult:
    """Riemannian proxy search using the transformed VAR sequence."""
    sequence, factor, identity_coordinates = _single_var_proxy_problem(system)
    initial = _projection_to_whitened(
        initial_projection,
        factor,
        identity_coordinates=identity_coordinates,
    ).to(dtype=sequence.dtype, device=sequence.device)
    result = _riemannian_armijo(
        initial,
        objective=lambda matrix: proxy_dynamical_dependence(matrix, sequence),
        gradient=lambda matrix: proxy_dynamical_dependence_gradient(
            matrix, sequence
        ),
        **options,
    )
    return _restore_physical_coordinates(
        result, factor, identity_coordinates=identity_coordinates
    )


def grassmann_distance_matrix(projection: torch.Tensor) -> torch.Tensor:
    r"""Return SSDI ``gmetrics`` distances for row-orthonormal subspaces.

    The metric is the largest principal angle divided by :math:`\pi/2`.  For
    small angles the residual/``asin`` formulation is used to avoid the loss of
    precision of a raw ``acos`` close to one, following SSDI ``subspacea``.
    """
    basis = torch.as_tensor(projection)
    if basis.ndim != 3:
        raise ValueError("projection must have shape (runs,m,n)")
    if basis.shape[0] < 1:
        raise ValueError("projection must contain at least one run")
    basis = orthonormalise_projection(basis)
    columns = basis.transpose(-1, -2)  # (runs,n,m)
    cross = torch.einsum("ami,bni->abmn", basis, basis)
    singular = torch.linalg.svdvals(cross).clamp(0.0, 1.0)
    cosine_min = singular[..., -1]
    angle_acos = torch.acos(cosine_min)

    approximation = torch.einsum("aim,abmk->abik", columns, cross)
    residual = columns.unsqueeze(0) - approximation
    sine_max = torch.linalg.svdvals(residual)[..., 0].clamp(0.0, 1.0)
    angle_asin = torch.asin(sine_max)
    angle = torch.where(
        cosine_min > math.sqrt(0.5), angle_asin, angle_acos
    )
    distance = angle / (math.pi / 2.0)
    distance = 0.5 * (distance + distance.transpose(-1, -2))
    distance.fill_diagonal_(0.0)
    return distance


def lcluster(
    distance: torch.Tensor,
    tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the exact sorted greedy semantics of MATLAB SSDI ``Lcluster``.

    The input endpoints must already be sorted by ascending proxy DD.  The
    first still-available endpoint becomes a cluster representative and every
    still-available endpoint at distance below ``tolerance`` is consumed by
    that cluster.  This is intentionally *not* replaced by generic linkage or
    k-means clustering.
    """
    matrix = torch.as_tensor(distance)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distance must be a square pairwise matrix")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("cluster_tolerance must be finite and positive")
    available = torch.ones(
        matrix.shape[0], dtype=torch.bool, device=matrix.device
    )
    representatives: list[int] = []
    sizes: list[int] = []
    for index in range(matrix.shape[0]):
        if not bool(available[index].item()):
            continue
        representatives.append(index)
        available[index] = False
        members = available & (matrix[index] < tolerance)
        sizes.append(1 + int(torch.count_nonzero(members).item()))
        available = available & ~members
    return (
        torch.tensor(representatives, dtype=torch.long, device=matrix.device),
        torch.tensor(sizes, dtype=torch.long, device=matrix.device),
    )


def _random_initial_projection(
    system: Model,
    *,
    output_dimension: int,
    restarts: int,
    seed: int | None,
) -> torch.Tensor:
    """Generate random row-orthonormal initial subspaces on the model device."""
    iss = _single_innovations(system)
    n = int(iss.observation.shape[-2])
    m = int(output_dimension)
    restarts = int(restarts)
    if not 1 <= m < n:
        raise ValueError("output_dimension must satisfy 1 <= m < n_variables")
    if restarts < 1:
        raise ValueError("restarts must be at least 1")
    generator = None
    if seed is not None:
        generator = torch.Generator(device=iss.observation.device)
        generator.manual_seed(int(seed))
    raw = torch.randn(
        (restarts, m, n),
        dtype=iss.observation.dtype,
        device=iss.observation.device,
        generator=generator,
    )
    return orthonormalise_projection(raw)


def _stage_options(
    *,
    optimizer: DDOptimizer,
    max_iterations: int | None,
    history: bool,
    common: Mapping[str, Any] | None,
    stage_specific: Mapping[str, Any] | None,
    spectral: bool,
) -> dict[str, Any]:
    """Resolve SSDI stage defaults and user overrides."""
    reserved = {"max_iterations", "history", "lags", "frequencies"}
    options = {} if common is None else dict(common)
    overlap = reserved.intersection(options)
    if overlap:
        names = ", ".join(sorted(overlap))
        raise ValueError(
            f"optimizer_options must not override common argument(s): {names}"
        )
    if optimizer == "complexbox":
        # MATLAB/ComplexBox often use variant 2.  ComplexTorch stages default
        # to variant 1 because its returned endpoint always corresponds to its
        # reported objective; variant 2 remains available explicitly for
        # literal reference-behaviour studies.
        resolved: dict[str, Any] = {
            "variant": 1,
            "initial_step_size": 0.1 if spectral else 1.0,
            "gdls": 2.0,
            "tol": 1e-10 if spectral else 1e-8,
        }
    else:
        resolved = {
            "initial_step_size": 1.0,
            "armijo_constant": 1e-4,
            "backtrack_factor": 0.5,
            "max_backtracks": 25,
            "min_step": 1e-12,
            "gradient_tolerance": 1e-9,
            "objective_tolerance": 1e-12,
        }
    resolved.update(options)
    if stage_specific is not None:
        overlap = {"max_iterations", "history"}.intersection(stage_specific)
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(
                "stage-specific optimizer options must not override common "
                f"argument(s): {names}"
            )
        resolved.update(dict(stage_specific))
    resolved["max_iterations"] = (
        10_000 if max_iterations is None else int(max_iterations)
    )
    resolved["history"] = history
    return resolved


def optimise_ssdi_workflow(
    system: Model,
    initial_projection: torch.Tensor | None,
    *,
    optimizer: DDOptimizer,
    output_dimension: int | None,
    restarts: int,
    lags: int | None,
    frequencies: torch.Tensor | None,
    frequency_bins: int,
    cluster_tolerance: float,
    max_iterations: int | None,
    history: bool,
    optimizer_options: Mapping[str, Any] | None,
    preoptimization_options: Mapping[str, Any] | None,
    spectral_options: Mapping[str, Any] | None,
    seed: int | None,
) -> SSDIRawResult:
    """Run proxy pre-optimisation, ``Lcluster``, then spectral refinement."""
    if initial_projection is None:
        if output_dimension is None:
            raise ValueError(
                "output_dimension is required when initial_projection is omitted"
            )
        initial = _random_initial_projection(
            system,
            output_dimension=output_dimension,
            restarts=restarts,
            seed=seed,
        )
    else:
        initial = torch.as_tensor(initial_projection)
        if initial.ndim not in (2, 3):
            raise ValueError(
                "initial_projection must have shape (m,n) or (runs,m,n)"
            )
        if (
            output_dimension is not None
            and int(output_dimension) != int(initial.shape[-2])
        ):
            raise ValueError(
                "output_dimension must match initial_projection output dimension"
            )
        initial = orthonormalise_projection(initial)

    pre_options = _stage_options(
        optimizer=optimizer,
        max_iterations=max_iterations,
        history=history,
        common=optimizer_options,
        stage_specific=preoptimization_options,
        spectral=False,
    )
    if isinstance(system, VARSystem):
        if lags is not None:
            raise ValueError(
                "lags is not configurable for staged VAR SSDI: the complete "
                "VAR coefficient sequence is the proxy CAK sequence"
            )
        if optimizer == "complexbox":
            pre = _optimise_var_proxy_complexbox(system, initial, **pre_options)
        else:
            pre = _optimise_var_proxy_riemannian(system, initial, **pre_options)
    else:
        if lags is not None:
            pre_options["lags"] = lags
        if optimizer == "complexbox":
            pre = optimise_dynamical_dependence_proxy(
                system, initial, **pre_options
            )
        else:
            pre = optimise_dynamical_dependence_proxy_riemannian(
                system, initial, **pre_options
            )

    distances = grassmann_distance_matrix(pre.projection)
    representatives, cluster_sizes = lcluster(distances, cluster_tolerance)
    spectral_initial = pre.projection[representatives]

    if frequencies is None:
        bins = int(frequency_bins)
        if bins < 2:
            raise ValueError("frequency_bins must be at least 2")
        iss = _single_innovations(system)
        frequencies = torch.linspace(
            0.0,
            0.5,
            bins,
            dtype=iss.transition.dtype,
            device=iss.transition.device,
        )
    else:
        frequencies = torch.as_tensor(frequencies)
        if frequencies.ndim != 1 or frequencies.numel() < 2:
            raise ValueError(
                "frequencies must be a one-dimensional grid with >=2 bins"
            )

    spectral_stage_options = _stage_options(
        optimizer=optimizer,
        max_iterations=max_iterations,
        history=history,
        common=optimizer_options,
        stage_specific=spectral_options,
        spectral=True,
    )
    if optimizer == "complexbox":
        spectral = optimise_dynamical_dependence_spectral(
            system,
            spectral_initial,
            frequencies,
            **spectral_stage_options,
        )
    else:
        spectral = optimise_dynamical_dependence_spectral_riemannian(
            system,
            spectral_initial,
            frequencies,
            **spectral_stage_options,
        )
    return SSDIRawResult(
        preoptimization=pre,
        spectral=spectral,
        cluster_indices=representatives,
        cluster_sizes=cluster_sizes,
        cluster_distances=distances,
    )


__all__ = [
    "SSDIRawResult",
    "grassmann_distance_matrix",
    "lcluster",
    "optimise_ssdi_workflow",
]
