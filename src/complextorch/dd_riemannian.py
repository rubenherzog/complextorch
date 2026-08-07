"""Riemannian optimization extensions for dynamical dependence.

The ComplexBox-compatible implementation in :mod:`complextorch.dd_optimization`
is intentionally left unchanged and serves as the frozen baseline. This module
adds two independent facilities around that baseline:

1. Torch autograd oracles for the proxy and spectral Grassmann gradients.
2. Riemannian gradient descent with Armijo backtracking on the row-Grassmann
   representation used by ComplexTorch.

The optimizer itself uses the audited analytic gradients from the frozen
baseline; autograd is only an independent gradient oracle for tests/audits.
The model-level wrappers reuse the exact innovation-whitening coordinate map
from ``dd_optimization``. Consequently, changing optimizer does not change the
objective, the physical macro-subspace, or the ``V > 0`` semantics.

References
----------
- Absil, Mahony, and Sepulchre (2008), *Optimization Algorithms on Matrix
  Manifolds*.
- Barnett and Seth (2023), dynamical-independence / SSDI formulation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .control import innovations_transfer_function
from .dd_optimization import (
    Model,
    _projection_from_whitened,
    _projection_to_whitened,
    _single_innovations,
    _whiten_innovations,
    innovations_proxy_sequence,
    orthonormalise_projection,
    proxy_dynamical_dependence,
    proxy_dynamical_dependence_gradient,
    spectral_dynamical_dependence,
    spectral_dynamical_dependence_gradient,
)


@dataclass(frozen=True)
class DDRiemannianSearchResult:
    """Result of Riemannian gradient descent with Armijo backtracking.

    Attributes
    ----------
    objective
        Sorted final objective values, shape ``(runs,)``.
    projection
        Corresponding row-orthonormal projections in physical observation
        coordinates, shape ``(runs,m,n)``.
    convergence
        Convergence codes: 0 maximum iterations, 1 gradient tolerance,
        2 objective-change tolerance, 3 Armijo step fell below ``min_step``.
    step_size
        Last accepted (or attempted on line-search failure) Armijo step.
    iterations
        Number of optimizer states including the initial state. This matches
        the iteration-count convention of the frozen ComplexBox baseline.
    objective_evaluations
        Number of scalar objective evaluations per run, including rejected
        Armijo candidates and the initial value.
    gradient_evaluations
        Number of Riemannian gradient evaluations per run.
    backtracking_evaluations
        Number of rejected Armijo candidate evaluations per run.
    history
        Optional ``(runs,max_iterations,4)`` tensor containing objective,
        accepted step size, gradient norm, and cumulative objective
        evaluations. Unused entries are NaN.
    """

    objective: torch.Tensor
    projection: torch.Tensor
    convergence: torch.Tensor
    step_size: torch.Tensor
    iterations: torch.Tensor
    objective_evaluations: torch.Tensor
    gradient_evaluations: torch.Tensor
    backtracking_evaluations: torch.Tensor
    history: torch.Tensor | None = None


def _grassmann_project_row(
    projection: torch.Tensor, euclidean: torch.Tensor
) -> torch.Tensor:
    r"""Project an ambient row-matrix gradient onto the Grassmann tangent.

    For a row-orthonormal representative ``M`` (``M M^T = I``), the horizontal
    Grassmann gradient is

    ``G = G_e - (G_e M^T) M = G_e (I - M^T M)``.

    This removes basis-rotation directions and matches the geometry optimized
    by the analytic SSDI gradients.
    """
    return euclidean - (euclidean @ projection.transpose(-1, -2)) @ projection


def _autograd_grassmann_gradient(
    projection: torch.Tensor,
    objective: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiate an objective and return its row-Grassmann gradient."""
    matrix = torch.as_tensor(projection)
    single = matrix.ndim == 2
    if single:
        matrix = matrix.unsqueeze(0)
    if matrix.ndim != 3:
        raise ValueError("projection must have shape (m,n) or (runs,m,n)")
    if not matrix.is_floating_point():
        raise TypeError("projection must use a floating-point dtype")

    with torch.enable_grad():
        variable = matrix.detach().clone().requires_grad_(True)
        value = objective(variable)
        if value.ndim == 0:
            value = value.unsqueeze(0)
        euclidean = torch.autograd.grad(value.sum(), variable)[0]
    gradient = _grassmann_project_row(variable.detach(), euclidean.detach())
    magnitude = torch.linalg.vector_norm(gradient, dim=(-2, -1))
    if single:
        return gradient[0], magnitude[0]
    return gradient, magnitude


def proxy_dynamical_dependence_autograd_gradient(
    projection: torch.Tensor,
    sequence: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Autograd oracle for the proxy-DD Grassmann gradient."""
    return _autograd_grassmann_gradient(
        projection,
        lambda matrix: proxy_dynamical_dependence(matrix, sequence),
    )


def spectral_dynamical_dependence_autograd_gradient(
    projection: torch.Tensor,
    transfer: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Autograd oracle for the spectral-DD Grassmann gradient."""
    return _autograd_grassmann_gradient(
        projection,
        lambda matrix: spectral_dynamical_dependence(matrix, transfer),
    )


def _validate_armijo_parameters(
    *,
    max_iterations: int,
    initial_step_size: float,
    armijo_constant: float,
    backtrack_factor: float,
    max_backtracks: int,
    min_step: float,
    gradient_tolerance: float,
    objective_tolerance: float,
) -> None:
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    if initial_step_size <= 0.0:
        raise ValueError("initial_step_size must be positive")
    if not 0.0 < armijo_constant < 1.0:
        raise ValueError("armijo_constant must lie strictly between 0 and 1")
    if not 0.0 < backtrack_factor < 1.0:
        raise ValueError("backtrack_factor must lie strictly between 0 and 1")
    if max_backtracks < 1:
        raise ValueError("max_backtracks must be at least 1")
    if min_step <= 0.0:
        raise ValueError("min_step must be positive")
    if gradient_tolerance < 0.0 or objective_tolerance < 0.0:
        raise ValueError("tolerances must be non-negative")


def _riemannian_armijo(
    initial_projection: torch.Tensor,
    *,
    objective: Callable[[torch.Tensor], torch.Tensor],
    gradient: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    max_iterations: int,
    initial_step_size: float,
    armijo_constant: float,
    backtrack_factor: float,
    max_backtracks: int,
    min_step: float,
    gradient_tolerance: float,
    objective_tolerance: float,
    history: bool,
) -> DDRiemannianSearchResult:
    """Run independent Armijo searches in one batched Torch state.

    Restarts are independent mathematically but evaluated together whenever
    they are at the same optimizer stage. During backtracking, boolean masks
    select only runs that still need a candidate evaluation. Runs that accept,
    converge, or fail line search leave the active batch without creating any
    transitions or state coupling between restarts.
    """
    _validate_armijo_parameters(
        max_iterations=max_iterations,
        initial_step_size=initial_step_size,
        armijo_constant=armijo_constant,
        backtrack_factor=backtrack_factor,
        max_backtracks=max_backtracks,
        min_step=min_step,
        gradient_tolerance=gradient_tolerance,
        objective_tolerance=objective_tolerance,
    )
    matrix = torch.as_tensor(initial_projection)
    if matrix.ndim == 2:
        matrix = matrix.unsqueeze(0)
    if matrix.ndim != 3 or matrix.shape[0] < 1:
        raise ValueError(
            "initial_projection must have shape (m,n) or (runs,m,n)"
        )
    matrix = orthonormalise_projection(matrix)
    nruns = matrix.shape[0]

    value = objective(matrix)
    if value.ndim == 0:
        value = value.unsqueeze(0)
    grad, grad_norm = gradient(matrix)
    if grad.ndim == 2:
        grad = grad.unsqueeze(0)
        grad_norm = grad_norm.unsqueeze(0)

    objective_evaluations = torch.ones(
        nruns, dtype=torch.int64, device=matrix.device
    )
    gradient_evaluations = torch.ones_like(objective_evaluations)
    rejected_evaluations = torch.zeros_like(objective_evaluations)
    step_size = torch.full(
        (nruns,), float(initial_step_size), dtype=matrix.dtype, device=matrix.device
    )
    convergence = torch.zeros(nruns, dtype=torch.int64, device=matrix.device)
    iterations = torch.ones(nruns, dtype=torch.int64, device=matrix.device)
    active = torch.ones(nruns, dtype=torch.bool, device=matrix.device)

    hist = None
    if history:
        hist = torch.full(
            (nruns, max_iterations, 4),
            torch.nan,
            dtype=matrix.dtype,
            device=matrix.device,
        )
        hist[:, 0, :] = torch.stack(
            (
                value,
                step_size,
                grad_norm,
                objective_evaluations.to(matrix.dtype),
            ),
            dim=-1,
        )

    with torch.no_grad():
        for state_index in range(2, max_iterations + 1):
            gradient_stopped = active & (grad_norm <= gradient_tolerance)
            convergence = torch.where(
                gradient_stopped,
                torch.ones_like(convergence),
                convergence,
            )
            active = active & ~gradient_stopped
            if not bool(torch.any(active).item()):
                break

            working = active.clone()
            previous = value.clone()
            squared_norm = grad_norm * grad_norm
            trial_step = step_size.clone()
            pending = working.clone()
            accepted = torch.zeros_like(active)
            candidate_matrix = matrix.clone()
            candidate_value = value.clone()

            # Evaluate all runs needing the same backtracking stage together.
            for _ in range(max_backtracks):
                if not bool(torch.any(pending).item()):
                    break
                indices = torch.nonzero(pending, as_tuple=False).flatten()
                trial = orthonormalise_projection(
                    matrix[indices]
                    - trial_step[indices, None, None] * grad[indices]
                )
                trial_value = objective(trial)
                if trial_value.ndim == 0:
                    trial_value = trial_value.unsqueeze(0)
                objective_evaluations[indices] += 1
                bound = (
                    value[indices]
                    - armijo_constant
                    * trial_step[indices]
                    * squared_norm[indices]
                )
                local_accept = torch.isfinite(trial_value) & (trial_value <= bound)
                accepted_indices = indices[local_accept]
                rejected_indices = indices[~local_accept]

                if accepted_indices.numel():
                    candidate_matrix[accepted_indices] = trial[local_accept]
                    candidate_value[accepted_indices] = trial_value[local_accept]
                    accepted[accepted_indices] = True
                    pending[accepted_indices] = False

                if rejected_indices.numel():
                    rejected_evaluations[rejected_indices] += 1
                    trial_step[rejected_indices] = (
                        trial_step[rejected_indices] * backtrack_factor
                    )
                    too_small = rejected_indices[
                        trial_step[rejected_indices] < min_step
                    ]
                    if too_small.numel():
                        pending[too_small] = False

            failed = working & ~accepted
            convergence = torch.where(
                failed,
                torch.full_like(convergence, 3),
                convergence,
            )
            active = active & ~failed
            step_size = torch.where(working, trial_step, step_size)

            if bool(torch.any(accepted).item()):
                indices = torch.nonzero(accepted, as_tuple=False).flatten()
                matrix[indices] = candidate_matrix[indices]
                value[indices] = candidate_value[indices]
                grad_new, grad_norm_new = gradient(matrix[indices])
                if grad_new.ndim == 2:
                    grad_new = grad_new.unsqueeze(0)
                    grad_norm_new = grad_norm_new.unsqueeze(0)
                grad[indices] = grad_new
                grad_norm[indices] = grad_norm_new
                gradient_evaluations[indices] += 1
                iterations[indices] = state_index

                if hist is not None:
                    hist[indices, state_index - 1, :] = torch.stack(
                        (
                            value[indices],
                            step_size[indices],
                            grad_norm[indices],
                            objective_evaluations[indices].to(matrix.dtype),
                        ),
                        dim=-1,
                    )

                scale = torch.maximum(
                    torch.ones_like(value[indices]),
                    torch.abs(previous[indices]),
                )
                local_objective_stopped = (
                    torch.abs(previous[indices] - value[indices])
                    <= objective_tolerance * scale
                )
                objective_stopped = torch.zeros_like(active)
                objective_stopped[indices] = local_objective_stopped
                convergence = torch.where(
                    objective_stopped,
                    torch.full_like(convergence, 2),
                    convergence,
                )
                active = active & ~objective_stopped

                # Match the single-run algorithm: only continuing runs propose
                # one larger step at the next state.
                continuing = accepted & ~objective_stopped
                step_size = torch.where(
                    continuing,
                    step_size / backtrack_factor,
                    step_size,
                )

            if not bool(torch.any(active).item()):
                break

    order = torch.argsort(value, stable=True)
    return DDRiemannianSearchResult(
        objective=value[order],
        projection=matrix[order],
        convergence=convergence[order],
        step_size=step_size[order],
        iterations=iterations[order],
        objective_evaluations=objective_evaluations[order],
        gradient_evaluations=gradient_evaluations[order],
        backtracking_evaluations=rejected_evaluations[order],
        history=None if hist is None else hist[order],
    )


def _restore_physical_coordinates(
    result: DDRiemannianSearchResult,
    factor: torch.Tensor,
    *,
    identity_coordinates: bool,
) -> DDRiemannianSearchResult:
    projection = _projection_from_whitened(
        result.projection,
        factor,
        identity_coordinates=identity_coordinates,
    )
    return DDRiemannianSearchResult(
        objective=result.objective,
        projection=projection,
        convergence=result.convergence,
        step_size=result.step_size,
        iterations=result.iterations,
        objective_evaluations=result.objective_evaluations,
        gradient_evaluations=result.gradient_evaluations,
        backtracking_evaluations=result.backtracking_evaluations,
        history=result.history,
    )


def optimise_dynamical_dependence_proxy_riemannian(
    system: Model,
    initial_projection: torch.Tensor,
    *,
    lags: int | None = None,
    max_iterations: int = 1_000,
    initial_step_size: float = 1.0,
    armijo_constant: float = 1e-4,
    backtrack_factor: float = 0.5,
    max_backtracks: int = 25,
    min_step: float = 1e-12,
    gradient_tolerance: float = 1e-9,
    objective_tolerance: float = 1e-12,
    history: bool = False,
) -> DDRiemannianSearchResult:
    """Minimize proxy DD by Riemannian gradient descent with Armijo search."""
    iss = _single_innovations(system)
    whitened, factor, identity_coordinates = _whiten_innovations(iss)
    sequence = innovations_proxy_sequence(whitened, lags=lags)
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
        max_iterations=max_iterations,
        initial_step_size=initial_step_size,
        armijo_constant=armijo_constant,
        backtrack_factor=backtrack_factor,
        max_backtracks=max_backtracks,
        min_step=min_step,
        gradient_tolerance=gradient_tolerance,
        objective_tolerance=objective_tolerance,
        history=history,
    )
    return _restore_physical_coordinates(
        result, factor, identity_coordinates=identity_coordinates
    )


def optimise_dynamical_dependence_spectral_riemannian(
    system: Model,
    initial_projection: torch.Tensor,
    frequencies: torch.Tensor,
    *,
    max_iterations: int = 1_000,
    initial_step_size: float = 1.0,
    armijo_constant: float = 1e-4,
    backtrack_factor: float = 0.5,
    max_backtracks: int = 25,
    min_step: float = 1e-12,
    gradient_tolerance: float = 1e-9,
    objective_tolerance: float = 1e-12,
    history: bool = False,
) -> DDRiemannianSearchResult:
    """Minimize spectral DD by Riemannian gradient descent with Armijo search."""
    iss = _single_innovations(system)
    whitened, factor, identity_coordinates = _whiten_innovations(iss)
    frequencies = torch.as_tensor(
        frequencies,
        dtype=whitened.transition.dtype,
        device=whitened.transition.device,
    )
    transfer = innovations_transfer_function(whitened, frequencies)
    if transfer.ndim == 4:
        transfer = transfer[0]
    initial = _projection_to_whitened(
        initial_projection,
        factor,
        identity_coordinates=identity_coordinates,
    ).to(
        dtype=whitened.observation.dtype,
        device=whitened.observation.device,
    )
    result = _riemannian_armijo(
        initial,
        objective=lambda matrix: spectral_dynamical_dependence(
            matrix, transfer
        ),
        gradient=lambda matrix: spectral_dynamical_dependence_gradient(
            matrix, transfer
        ),
        max_iterations=max_iterations,
        initial_step_size=initial_step_size,
        armijo_constant=armijo_constant,
        backtrack_factor=backtrack_factor,
        max_backtracks=max_backtracks,
        min_step=min_step,
        gradient_tolerance=gradient_tolerance,
        objective_tolerance=objective_tolerance,
        history=history,
    )
    return _restore_physical_coordinates(
        result, factor, identity_coordinates=identity_coordinates
    )


__all__ = [
    "DDRiemannianSearchResult",
    "optimise_dynamical_dependence_proxy_riemannian",
    "optimise_dynamical_dependence_spectral_riemannian",
    "proxy_dynamical_dependence_autograd_gradient",
    "spectral_dynamical_dependence_autograd_gradient",
]
