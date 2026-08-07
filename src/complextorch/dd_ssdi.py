"""Staged SSDI dynamical-independence optimisation workflow.

This internal module implements the algorithmic process used by the Barnett--Seth
SSDI MATLAB toolbox: proxy-DD pre-optimisation over many random restarts,
Grassmann clustering of the proxy minima, and full spectral-DD refinement from
one representative per cluster. Linear algebra remains native Torch and uses
ComplexTorch's row-projection convention.

The module intentionally contains orchestration only. Proxy/spectral objectives,
gradients, innovation whitening, and the two optimiser families remain in their
existing modules.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import torch

from .control import InnovationsStateSpace
from .dd_optimization import (
    DDGradientSearchResult,
    Model,
    _optimise,
    _projection_from_whitened,
    _projection_to_whitened,
    _single_innovations,
    optimise_dynamical_dependence_proxy,
    optimise_dynamical_dependence_spectral,
    orthonormalise_projection,
    proxy_dynamical_dependence,
    proxy_dynamical_dependence_gradient,
)
from .dd_riemannian import (
    DDRiemannianSearchResult,
    _riemannian_armijo,
    optimise_dynamical_dependence_proxy_riemannian,
    optimise_dynamical_dependence_spectral_riemannian,
)
from .representations import VARSystem

Optimizer = Literal["complexbox", "riemannian_armijo"]
RawSearchResult = DDGradientSearchResult | DDRiemannianSearchResult


@dataclass(frozen=True)
class SSDIWorkflowRawResult:
    """Internal result of the two-stage SSDI workflow."""

    preoptimization: RawSearchResult
    cluster_representative_indices: torch.Tensor
    cluster_sizes: torch.Tensor
    cluster_distances: torch.Tensor
    spectral: RawSearchResult
    frequencies: torch.Tensor


def grassmann_distances(projection: torch.Tensor) -> torch.Tensor:
    r"""Return SSDI ``gmetrics`` normalized maximum-principal-angle distances.

    Parameters
    ----------
    projection
        Row-orthonormal subspaces with shape ``(runs, m, n)``.

    Returns
    -------
    torch.Tensor
        Symmetric matrix ``(runs, runs)`` with entries in ``[0, 1]``. The
        normalization is by :math:`\pi/2`, matching SSDI ``gmetric(...,true)``.
    """
    matrix = torch.as_tensor(projection)
    if matrix.ndim != 3:
        raise ValueError("projection must have shape (runs,m,n)")
    matrix = orthonormalise_projection(matrix)
    cross = torch.einsum("ami,bni->abmn", matrix, matrix)
    singular = torch.linalg.svdvals(cross)
    smallest = singular[..., -1].clamp(0.0, 1.0)
    distance = torch.acos(smallest) / (torch.pi / 2.0)
    distance = 0.5 * (distance + distance.transpose(0, 1))
    distance.fill_diagonal_(0.0)
    return distance


def lcluster(
    distances: torch.Tensor,
    sorted_objective: torch.Tensor,
    *,
    tolerance: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cluster sorted proxy minima with MATLAB SSDI ``Lcluster`` semantics."""
    dist = torch.as_tensor(distances)
    dd = torch.as_tensor(sorted_objective)
    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
        raise ValueError("distances must be a square matrix")
    if dd.ndim != 1 or dd.shape[0] != dist.shape[0]:
        raise ValueError("sorted_objective must match distances")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if dd.numel() > 1 and bool(torch.any(dd[1:] < dd[:-1]).item()):
        raise ValueError("sorted_objective must be ascending")

    available = torch.ones(dist.shape[0], dtype=torch.bool, device=dist.device)
    representatives: list[int] = []
    sizes: list[int] = []
    for i in range(dist.shape[0]):
        if not bool(available[i].item()):
            continue
        representatives.append(i)
        members = available & (dist[i] < tolerance)
        members[i] = True
        sizes.append(int(members.sum().item()))
        available = available & ~members

    return (
        torch.tensor(representatives, dtype=torch.long, device=dist.device),
        torch.tensor(sizes, dtype=torch.long, device=dist.device),
    )


def _single_var(system: VARSystem) -> tuple[torch.Tensor, torch.Tensor]:
    coefficients = torch.as_tensor(system.coefficients)
    covariance = torch.as_tensor(system.innovation_covariance)
    if coefficients.ndim == 4:
        if coefficients.shape[0] != 1:
            raise ValueError("SSDI optimization currently accepts one microscopic VAR system")
        coefficients = coefficients[0]
    if covariance.ndim == 3:
        if covariance.shape[0] != 1:
            raise ValueError("SSDI optimization currently accepts one microscopic VAR system")
        covariance = covariance[0]
    if coefficients.ndim != 3:
        raise ValueError("VAR coefficients must have shape (p,n,n)")
    return coefficients, covariance


def _var_proxy_inputs(
    system: VARSystem,
    initial_projection: torch.Tensor,
    *,
    lags: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Return whitened VAR coefficients and transformed initial subspaces.

    MATLAB SSDI ``cak2ddx`` accepts the VAR coefficient sequence directly. For
    ``V=B B^T`` the exact residual-normalising transform is
    ``A_k <- B^-1 A_k B`` and physical row projections map as ``M <- M B``.
    """
    coefficients, covariance = _single_var(system)
    p, n, _ = coefficients.shape
    if lags is None:
        lags = p
    lags = int(lags)
    if not 1 <= lags <= p:
        raise ValueError("for VAR SSDI proxy optimization, lags must be in [1,p]")
    coefficients = coefficients[:lags]
    identity = torch.eye(n, dtype=covariance.dtype, device=covariance.device)
    identity_coordinates = bool(torch.allclose(covariance, identity, rtol=1e-7, atol=1e-9))
    if identity_coordinates:
        factor = identity
        sequence = coefficients
    else:
        try:
            factor = torch.linalg.cholesky(covariance)
        except RuntimeError as exc:
            raise ValueError("innovation covariance must be positive definite") from exc
        right = coefficients @ factor
        factors = factor.expand(coefficients.shape[0], -1, -1)
        sequence = torch.linalg.solve_triangular(factors, right, upper=False)
    initial = _projection_to_whitened(
        initial_projection,
        factor,
        identity_coordinates=identity_coordinates,
    ).to(dtype=sequence.dtype, device=sequence.device)
    return sequence, initial, factor, identity_coordinates


def _complexbox_proxy_var(
    system: VARSystem,
    initial_projection: torch.Tensor,
    *,
    lags: int | None,
    max_iterations: int,
    history: bool,
    options: Mapping[str, Any],
) -> DDGradientSearchResult:
    sequence, initial, factor, identity_coordinates = _var_proxy_inputs(
        system, initial_projection, lags=lags
    )
    opts = dict(options)
    variant = int(opts.pop("variant", 1))
    initial_step_size = float(opts.pop("initial_step_size", 1e-3))
    gdls = opts.pop("gdls", 2.0)
    tol = opts.pop("tol", 1e-9)
    if opts:
        raise TypeError(f"unknown ComplexBox optimizer option(s): {sorted(opts)}")
    raw = _optimise(
        initial,
        objective=lambda matrix: proxy_dynamical_dependence(matrix, sequence),
        gradient=lambda matrix: proxy_dynamical_dependence_gradient(matrix, sequence),
        max_iterations=max_iterations,
        variant=variant,
        initial_step_size=initial_step_size,
        gdls=gdls,
        tol=tol,
        spectral=False,
        history=history,
    )
    return DDGradientSearchResult(
        objective=raw.objective,
        projection=_projection_from_whitened(
            raw.projection, factor, identity_coordinates=identity_coordinates
        ),
        convergence=raw.convergence,
        step_size=raw.step_size,
        iterations=raw.iterations,
        history=raw.history,
    )


def _riemannian_proxy_var(
    system: VARSystem,
    initial_projection: torch.Tensor,
    *,
    lags: int | None,
    max_iterations: int,
    history: bool,
    options: Mapping[str, Any],
) -> DDRiemannianSearchResult:
    sequence, initial, factor, identity_coordinates = _var_proxy_inputs(
        system, initial_projection, lags=lags
    )
    opts = dict(options)
    kwargs = {
        "initial_step_size": opts.pop("initial_step_size", 1.0),
        "armijo_constant": opts.pop("armijo_constant", 1e-4),
        "backtrack_factor": opts.pop("backtrack_factor", 0.5),
        "max_backtracks": opts.pop("max_backtracks", 25),
        "min_step": opts.pop("min_step", 1e-12),
        "gradient_tolerance": opts.pop("gradient_tolerance", 1e-9),
        "objective_tolerance": opts.pop("objective_tolerance", 1e-12),
    }
    if opts:
        raise TypeError(f"unknown Riemannian optimizer option(s): {sorted(opts)}")
    raw = _riemannian_armijo(
        initial,
        objective=lambda matrix: proxy_dynamical_dependence(matrix, sequence),
        gradient=lambda matrix: proxy_dynamical_dependence_gradient(matrix, sequence),
        max_iterations=max_iterations,
        history=history,
        **kwargs,
    )
    projection = _projection_from_whitened(
        raw.projection, factor, identity_coordinates=identity_coordinates
    )
    return DDRiemannianSearchResult(
        objective=raw.objective,
        projection=projection,
        convergence=raw.convergence,
        step_size=raw.step_size,
        iterations=raw.iterations,
        objective_evaluations=raw.objective_evaluations,
        gradient_evaluations=raw.gradient_evaluations,
        backtracking_evaluations=raw.backtracking_evaluations,
        history=raw.history,
    )


def _proxy_stage(
    system: Model,
    initial_projection: torch.Tensor,
    *,
    optimizer: Optimizer,
    lags: int | None,
    max_iterations: int,
    history: bool,
    options: Mapping[str, Any],
) -> RawSearchResult:
    if isinstance(system, VARSystem):
        if optimizer == "complexbox":
            return _complexbox_proxy_var(
                system,
                initial_projection,
                lags=lags,
                max_iterations=max_iterations,
                history=history,
                options=options,
            )
        return _riemannian_proxy_var(
            system,
            initial_projection,
            lags=lags,
            max_iterations=max_iterations,
            history=history,
            options=options,
        )

    opts = dict(options)
    if optimizer == "complexbox":
        opts.setdefault("variant", 1)
        return optimise_dynamical_dependence_proxy(
            system,
            initial_projection,
            lags=lags,
            max_iterations=max_iterations,
            history=history,
            **opts,
        )
    return optimise_dynamical_dependence_proxy_riemannian(
        system,
        initial_projection,
        lags=lags,
        max_iterations=max_iterations,
        history=history,
        **opts,
    )


def _spectral_stage(
    system: Model,
    initial_projection: torch.Tensor,
    frequencies: torch.Tensor,
    *,
    optimizer: Optimizer,
    max_iterations: int,
    history: bool,
    options: Mapping[str, Any],
) -> RawSearchResult:
    opts = dict(options)
    if optimizer == "complexbox":
        opts.setdefault("variant", 1)
        return optimise_dynamical_dependence_spectral(
            system,
            initial_projection,
            frequencies,
            max_iterations=max_iterations,
            history=history,
            **opts,
        )
    return optimise_dynamical_dependence_spectral_riemannian(
        system,
        initial_projection,
        frequencies,
        max_iterations=max_iterations,
        history=history,
        **opts,
    )


def default_frequency_grid(system: Model, points: int = 513) -> torch.Tensor:
    """Return the SSDI one-sided frequency grid over ``[0, 0.5]``."""
    if points < 2:
        raise ValueError("frequency_points must be at least 2")
    iss: InnovationsStateSpace = _single_innovations(system)
    return torch.linspace(
        0.0,
        0.5,
        points,
        dtype=iss.transition.dtype,
        device=iss.transition.device,
    )


def run_ssdi_workflow(
    system: Model,
    initial_projection: torch.Tensor,
    *,
    optimizer: Optimizer = "complexbox",
    lags: int | None = None,
    frequencies: torch.Tensor | None = None,
    frequency_points: int = 513,
    cluster_tolerance: float = 0.01,
    preoptimization_max_iterations: int = 10_000,
    spectral_max_iterations: int = 10_000,
    history: bool = False,
    preoptimizer_options: Mapping[str, Any] | None = None,
    spectral_optimizer_options: Mapping[str, Any] | None = None,
) -> SSDIWorkflowRawResult:
    """Execute proxy pre-optimization, ``Lcluster``, and spectral refinement."""
    if optimizer not in ("complexbox", "riemannian_armijo"):
        raise ValueError("optimizer must be 'complexbox' or 'riemannian_armijo'")
    if preoptimization_max_iterations < 1 or spectral_max_iterations < 1:
        raise ValueError("iteration limits must be positive")
    initial = torch.as_tensor(initial_projection)
    if initial.ndim != 3 or initial.shape[0] < 1:
        raise ValueError("staged SSDI requires initial_projection shape (runs,m,n)")

    pre = _proxy_stage(
        system,
        initial,
        optimizer=optimizer,
        lags=lags,
        max_iterations=preoptimization_max_iterations,
        history=history,
        options={} if preoptimizer_options is None else preoptimizer_options,
    )
    distances = grassmann_distances(pre.projection)
    representatives, sizes = lcluster(
        distances, pre.objective, tolerance=cluster_tolerance
    )
    frequency_grid = (
        default_frequency_grid(system, frequency_points)
        if frequencies is None
        else torch.as_tensor(frequencies)
    )
    spectral_initial = pre.projection[representatives]
    spectral = _spectral_stage(
        system,
        spectral_initial,
        frequency_grid,
        optimizer=optimizer,
        max_iterations=spectral_max_iterations,
        history=history,
        options={} if spectral_optimizer_options is None else spectral_optimizer_options,
    )
    return SSDIWorkflowRawResult(
        preoptimization=pre,
        cluster_representative_indices=representatives,
        cluster_sizes=sizes,
        cluster_distances=distances,
        spectral=spectral,
        frequencies=frequency_grid,
    )


__all__ = [
    "SSDIWorkflowRawResult",
    "default_frequency_grid",
    "grassmann_distances",
    "lcluster",
    "run_ssdi_workflow",
]
