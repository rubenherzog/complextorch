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


def _riemannian_armijo_single(
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
    """Optimize one row-Grassmann representative with Armijo backtracking."""
    matrix = torch.as_tensor(initial_projection)
    if matrix.ndim != 2:
        raise ValueError("single-run projection must have shape (m,n)")
    matrix = orthonormalise_projection(matrix)

    value = objective(matrix)
    if value.ndim != 0:
        value = value.squeeze()
    grad, grad_norm = gradient(matrix)
    objective_evaluations = 1
    gradient_evaluations = 1
    rejected_evaluations = 0
    step_size = torch.as_tensor(
        initial_step_size, dtype=matrix.dtype, device=matrix.device
    )
    convergence = 0
    iterations = 1

    hist = None
    if history:
        hist = torch.full(
            (max_iterations, 4),
            torch.nan,
            dtype=matrix.dtype,
            device=matrix.device,
        )
        hist[0] = torch.stack(
            (
                value,
                step_size,
                grad_norm,
                torch.as_tensor(
                    float(objective_evaluations),
                    dtype=matrix.dtype,
                    device=matrix.device,
                ),
            )
        )

    with torch.no_grad():
        for state_index in range(2, max_iterations + 1):
            if bool(grad_norm <= gradient_tolerance):
                convergence = 1
                break

            previous = value
            squared_norm = grad_norm * grad_norm
            trial_step = step_size
            accepted = False
            candidate = matrix
            candidate_value = value

            for _ in range(max_backtracks):
                candidate = orthonormalise_projection(matrix - trial_step * grad)
                candidate_value = objective(candidate)
                if candidate_value.ndim != 0:
                    candidate_value = candidate_value.squeeze()
                objective_evaluations += 1
                armijo_bound = value - armijo_constant * trial_step * squared_norm
                if bool(
                    torch.isfinite(candidate_value)
                    & (candidate_value <= armijo_bound)
                ):
                    accepted = True
                    break
                rejected_evaluations += 1
                trial_step = trial_step * backtrack_factor
                if bool(trial_step < min_step):
                    break

            step_size = trial_step
            if not accepted:
                convergence = 3
                break

            matrix = candidate
            value = candidate_value
            grad, grad_norm = gradient(matrix)
            gradient_evaluations += 1
            iterations = state_index

            if hist is not None:
                hist[state_index - 1] = torch.stack(
                    (
                        value,
                        step_size,
                        grad_norm,
                        torch.as_tensor(
                            float(objective_evaluations),
                            dtype=matrix.dtype,
                            device=matrix.device,
                        ),
                    )
                )

            scale = torch.maximum(
                torch.ones((), dtype=value.dtype, device=value.device),
                torch.abs(previous),
            )
            if bool(
                torch.abs(previous - value) <= objective_tolerance * scale
            ):
                convergence = 2
                break

            # Try one larger step at the next state; Armijo will backtrack if
            # the local curvature does not support it.
            step_size = step_size / backtrack_factor

    return DDRiemannianSearchResult(
        objective=value.reshape(1),
        projection=matrix.unsqueeze(0),
        convergence=torch.tensor(
            [convergence], dtype=torch.int64, device=matrix.device
        ),
        step_size=step_size.reshape(1),
        iterations=torch.tensor(
            [iterations], dtype=torch.int64, device=matrix.device
        ),
        objective_evaluations=torch.tensor(
            [objective_evaluations], dtype=torch.int64, device=matrix.device
        ),
        gradient_evaluations=torch.tensor(
            [gradient_evaluations], dtype=torch.int64, device=matrix.device
        ),
        backtracking_evaluations=torch.tensor(
            [rejected_evaluations], dtype=torch.int64, device=matrix.device
        ),
        history=None if hist is None else hist.unsqueeze(0),
    )


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
    """Run independent Armijo searches and sort by final objective."""
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
    projection = torch.as_tensor(initial_projection)
    if projection.ndim == 2:
        projection = projection.unsqueeze(0)
    if projection.ndim != 3 or projection.shape[0] < 1:
        raise ValueError(
            "initial_projection must have shape (m,n) or (runs,m,n)"
        )

    runs = [
        _riemannian_armijo_single(
            projection[index],
            objective=objective,
            gradient=gradient,
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
        for index in range(projection.shape[0])
    ]

    objective_values = torch.cat([run.objective for run in runs])
    order = torch.argsort(objective_values, stable=True)
    history_values = None
    if history:
        history_values = torch.cat(
            [run.history for run in runs if run.history is not None]
        )

    return DDRiemannianSearchResult(
        objective=objective_values[order],
        projection=torch.cat([run.projection for run in runs], dim=0)[order],
        convergence=torch.cat([run.convergence for run in runs])[order],
        step_size=torch.cat([run.step_size for run in runs])[order],
        iterations=torch.cat([run.iterations for run in runs])[order],
        objective_evaluations=torch.cat(
            [run.objective_evaluations for run in runs]
        )[order],
        gradient_evaluations=torch.cat(
            [run.gradient_evaluations for run in runs]
        )[order],
        backtracking_evaluations=torch.cat(
            [run.backtracking_evaluations for run in runs]
        )[order],
        history=None if history_values is None else history_values[order],
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
