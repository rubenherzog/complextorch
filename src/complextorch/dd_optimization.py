"""Optimization of dynamical dependence over projection subspaces.

The canonical scientific workflow is staged SSDI: proxy multi-start search,
Grassmann clustering, and spectral refinement. ``adaptive`` and ``armijo`` are
numerical step policies for that same workflow.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping

import torch

from .control import InnovationsStateSpace, _as_innovations_state_space, innovations_transfer_function
from .representations import StateSpaceModel, VARSystem
DDObjective = Literal["proxy", "spectral"]
DDOptimizer = Literal["adaptive", "armijo", "complexbox", "riemannian_armijo"]

from .dd import (
    Model, _single_innovations, innovations_proxy_sequence,
    proxy_dynamical_dependence, proxy_dynamical_dependence_gradient,
    spectral_dynamical_dependence, spectral_dynamical_dependence_gradient,
)

@dataclass(frozen=True)
class DDGradientSearchResult:
    """Result of a batched SSDI gradient-descent search.

    Attributes
    ----------
    objective
        Sorted final objective values, shape ``(runs,)``.
    projection
        Corresponding row projections in the original observation coordinates,
        shape ``(runs, m, n)``.
    convergence
        SSDI convergence codes: 0 unconverged, 1 step size below tolerance,
        2 objective below tolerance, 3 gradient norm below tolerance.
    step_size
        Final step size for each sorted run.
    iterations
        Iteration at which each run stopped.
    history
        Optional tensor ``(runs, max_recorded_iterations, 3)`` containing
        ``(objective, step_size, gradient_norm)``. Entries after a run stops
        remain ``nan``.
    """

    objective: torch.Tensor
    projection: torch.Tensor
    convergence: torch.Tensor
    step_size: torch.Tensor
    iterations: torch.Tensor
    history: torch.Tensor | None = None


def orthonormalise_projection(projection: torch.Tensor) -> torch.Tensor:
    """SVD retraction matching ComplexBox ``orthonormalise``.

    ComplexBox orthonormalises a column basis ``L`` with the left singular
    vectors of ``L``. For ``M=L.T`` this is equivalent to the right singular
    vectors of ``M``.
    """
    matrix = torch.as_tensor(projection)
    single = matrix.ndim == 2
    if single:
        matrix = matrix.unsqueeze(0)
    if matrix.ndim != 3:
        raise ValueError("projection must be unbatched or batched")
    _, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    return vh[0] if single else vh


def _whiten_innovations(
    system: InnovationsStateSpace,
) -> tuple[InnovationsStateSpace, torch.Tensor, bool]:
    r"""Whiten a general innovations system without changing its process.

    Let ``V = B B^T`` be the lower Cholesky factorization and define

    .. math::

       z_t=B^{-1}y_t,\qquad \eta_t=B^{-1}\varepsilon_t.

    Then ``cov(eta)=I`` and the equivalent innovations model is

    .. math::

       x_{t+1}=A x_t + K B\eta_t,\qquad
       z_t=B^{-1}C x_t+\eta_t.

    Thus ``C_w=B^{-1}C`` and ``K_w=KB``. ``B`` is returned because it is
    also the exact coordinate map between physical and whitened projections.
    """
    covariance = torch.as_tensor(system.innovation_covariance)
    identity = torch.eye(
        covariance.shape[-1], dtype=covariance.dtype, device=covariance.device
    )
    is_identity = bool(torch.allclose(covariance, identity, rtol=1e-7, atol=1e-9))
    if is_identity:
        return system, identity, True
    try:
        factor = torch.linalg.cholesky(covariance)
    except RuntimeError as exc:
        raise ValueError("innovation covariance must be positive definite") from exc
    observation = torch.linalg.solve_triangular(
        factor, system.observation, upper=False
    )
    gain = system.gain @ factor
    whitened = InnovationsStateSpace(
        system.transition,
        observation,
        gain,
        identity,
    )
    return whitened, factor, False


def _projection_to_whitened(
    projection: torch.Tensor,
    factor: torch.Tensor,
    *,
    identity_coordinates: bool,
) -> torch.Tensor:
    r"""Map ``M y`` to an orthonormal basis of the same macroprocess in ``z``.

    Since ``y=Bz``, the physical projection ``M`` becomes ``M B`` in whitened
    coordinates. Orthonormalization changes only the macro coordinate basis,
    not the Grassmann subspace or dynamical dependence.
    """
    matrix = torch.as_tensor(
        projection, dtype=factor.dtype, device=factor.device
    )
    if identity_coordinates:
        return matrix
    return orthonormalise_projection(matrix @ factor)


def _projection_from_whitened(
    projection: torch.Tensor,
    factor: torch.Tensor,
    *,
    identity_coordinates: bool,
) -> torch.Tensor:
    r"""Map a whitened row subspace back to the original observation space.

    For ``y=Bz`` and whitened projection ``N z``, the corresponding physical
    row map is ``N B^{-1} y``. The right division is evaluated by a triangular
    solve; no explicit inverse is formed. The returned rows are orthonormalized
    only to choose a Stiefel representative of the same subspace.
    """
    matrix = torch.as_tensor(projection)
    if identity_coordinates:
        return matrix
    single = matrix.ndim == 2
    if single:
        matrix = matrix.unsqueeze(0)
    physical_t = torch.linalg.solve_triangular(
        factor.transpose(-1, -2),
        matrix.transpose(-1, -2),
        upper=True,
    )
    physical = orthonormalise_projection(physical_t.transpose(-1, -2))
    return physical[0] if single else physical


def _restore_result_coordinates(
    result: DDGradientSearchResult,
    factor: torch.Tensor,
    *,
    identity_coordinates: bool,
) -> DDGradientSearchResult:
    """Return an optimizer result with projections in physical coordinates."""
    projection = _projection_from_whitened(
        result.projection,
        factor,
        identity_coordinates=identity_coordinates,
    )
    return DDGradientSearchResult(
        objective=result.objective,
        projection=projection,
        convergence=result.convergence,
        step_size=result.step_size,
        iterations=result.iterations,
        history=result.history,
    )


def _parse_factors(gdls: float | tuple[float, float]) -> tuple[float, float]:
    """Parse ComplexBox's gradient-descent line-search factors."""
    if isinstance(gdls, (float, int)):
        ifac = float(gdls)
        if ifac <= 0.0:
            raise ValueError("gdls must be positive")
        return ifac, 1.0 / ifac
    if len(gdls) != 2:
        raise ValueError("gdls must be a scalar or a two-element tuple")
    ifac, nfac = map(float, gdls)
    if ifac <= 0.0 or nfac <= 0.0:
        raise ValueError("gdls factors must be positive")
    return ifac, nfac


def _parse_tolerance(
    tol: float | tuple[float, float, float], *, spectral: bool
) -> tuple[float, float, float]:
    """Parse SSDI's ``(step, objective, gradient)`` tolerances."""
    if isinstance(tol, (float, int)):
        value = float(tol)
        return value, value, value if spectral else value / 10.0
    if len(tol) != 3:
        raise ValueError("tol must be a scalar or a three-element tuple")
    return tuple(float(value) for value in tol)


def _optimise(
    initial_projection: torch.Tensor,
    *,
    objective: Callable[[torch.Tensor], torch.Tensor],
    gradient: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    max_iterations: int,
    variant: int,
    initial_step_size: float,
    gdls: float | tuple[float, float],
    tol: float | tuple[float, float, float],
    spectral: bool,
    history: bool,
) -> DDGradientSearchResult:
    """Batched port of ComplexBox ``_optimise_tensor_batch``."""
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    if variant not in (1, 2):
        raise ValueError("variant must be 1 or 2")
    if initial_step_size <= 0.0:
        raise ValueError("initial_step_size must be positive")
    ifac, nfac = _parse_factors(gdls)
    stol, dtol, gtol = _parse_tolerance(tol, spectral=spectral)

    projection = torch.as_tensor(initial_projection)
    if projection.ndim == 2:
        projection = projection.unsqueeze(0)
    if projection.ndim != 3:
        raise ValueError("initial_projection must have shape (m,n) or (runs,m,n)")
    l = projection.clone()
    nruns = l.shape[0]
    dd = objective(l)
    grad, gmag = gradient(l)
    sigma = torch.full(
        (nruns,), float(initial_step_size), dtype=l.dtype, device=l.device
    )
    active = torch.ones(nruns, dtype=torch.bool, device=l.device)
    convergence = torch.zeros(nruns, dtype=torch.int64, device=l.device)
    stop_iterations = torch.full(
        (nruns,), int(max_iterations), dtype=torch.int64, device=l.device
    )

    hist = None
    if history:
        hist = torch.full(
            (nruns, max_iterations, 3),
            torch.nan,
            dtype=l.dtype,
            device=l.device,
        )
        hist[:, 0, :] = torch.stack((dd, sigma, gmag), dim=-1)

    with torch.no_grad():
        for iteration in range(2, max_iterations + 1):
            safe = torch.where(gmag > 0.0, gmag, torch.ones_like(gmag))
            step = sigma[:, None, None] * grad / safe[:, None, None]
            step = torch.where(
                active[:, None, None], step, torch.zeros_like(step)
            )
            candidate_all = orthonormalise_projection(l - step)
            candidate = torch.where(active[:, None, None], candidate_all, l)

            if variant == 1:
                dd_try = objective(candidate)
                accept = active & (dd_try < dd)
                l = torch.where(accept[:, None, None], candidate, l)
                dd = torch.where(accept, dd_try, dd)
                sigma = torch.where(
                    active,
                    torch.where(accept, sigma * ifac, sigma * nfac),
                    sigma,
                )
                grad_try, gmag_try = gradient(l)
                grad = torch.where(accept[:, None, None], grad_try, grad)
                gmag = torch.where(accept, gmag_try, gmag)
            else:
                l = candidate
                grad_new, gmag_new = gradient(l)
                dd_new = objective(l)
                improve = active & (dd_new < dd)
                dd = torch.where(improve, dd_new, dd)
                sigma = torch.where(
                    active,
                    torch.where(improve, sigma * ifac, sigma * nfac),
                    sigma,
                )
                grad = torch.where(active[:, None, None], grad_new, grad)
                gmag = torch.where(active, gmag_new, gmag)

            if hist is not None:
                hist[:, iteration - 1, :] = torch.stack(
                    (dd, sigma, gmag), dim=-1
                )

            c1 = active & (sigma < stol)
            remaining = active & ~c1
            c2 = remaining & (dd < dtol)
            remaining = remaining & ~c2
            c3 = remaining & (gmag < gtol)
            stopped = c1 | c2 | c3
            convergence = torch.where(
                c1, torch.ones_like(convergence), convergence
            )
            convergence = torch.where(
                c2, torch.full_like(convergence, 2), convergence
            )
            convergence = torch.where(
                c3, torch.full_like(convergence, 3), convergence
            )
            stop_iterations = torch.where(
                stopped,
                torch.full_like(stop_iterations, iteration),
                stop_iterations,
            )
            active = active & ~stopped
            if not bool(torch.any(active).item()):
                break

    order = torch.argsort(dd, stable=True)
    return DDGradientSearchResult(
        objective=dd[order],
        projection=l[order],
        convergence=convergence[order],
        step_size=sigma[order],
        iterations=stop_iterations[order],
        history=None if hist is None else hist[order],
    )


def optimise_dynamical_dependence_proxy(
    system: Model,
    initial_projection: torch.Tensor,
    *,
    lags: int | None = None,
    max_iterations: int = 10_000,
    variant: int = 2,
    initial_step_size: float = 1e-3,
    gdls: float | tuple[float, float] = 2.0,
    tol: float | tuple[float, float, float] = 1e-9,
    history: bool = False,
) -> DDGradientSearchResult:
    r"""Optimize ComplexBox proxy DD for any positive-definite ``V``.

    General innovations are whitened exactly using ``V=B B^T``. If ``M`` is
    an initial physical projection, the optimizer is initialized on the
    equivalent whitened subspace ``row(MB)`` and the optimized subspace is
    mapped back through ``B^{-1}`` before being returned.
    """
    iss = _single_innovations(system)
    whitened, factor, identity_coordinates = _whiten_innovations(iss)
    sequence = innovations_proxy_sequence(whitened, lags=lags)
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
        max_iterations=max_iterations,
        variant=variant,
        initial_step_size=initial_step_size,
        gdls=gdls,
        tol=tol,
        spectral=False,
        history=history,
    )
    return _restore_result_coordinates(
        result,
        factor,
        identity_coordinates=identity_coordinates,
    )


def optimise_dynamical_dependence_spectral(
    system: Model,
    initial_projection: torch.Tensor,
    frequencies: torch.Tensor,
    *,
    max_iterations: int = 10_000,
    variant: int = 2,
    initial_step_size: float = 1e-3,
    gdls: float | tuple[float, float] = 2.0,
    tol: float | tuple[float, float, float] = 1e-9,
    history: bool = False,
) -> DDGradientSearchResult:
    r"""Optimize ComplexBox spectral DD for any positive-definite ``V``.

    The general ISS is transformed to the exactly equivalent identity-
    innovation coordinates before applying ``trfun2dd``/``trfun2ddgrad``.
    Returned projections are mapped back to the original observation space.
    """
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
    ).to(dtype=whitened.observation.dtype, device=whitened.observation.device)
    result = _optimise(
        initial,
        objective=lambda matrix: spectral_dynamical_dependence(
            matrix, transfer
        ),
        gradient=lambda matrix: spectral_dynamical_dependence_gradient(
            matrix, transfer
        ),
        max_iterations=max_iterations,
        variant=variant,
        initial_step_size=initial_step_size,
        gdls=gdls,
        tol=tol,
        spectral=True,
        history=history,
    )
    return _restore_result_coordinates(
        result,
        factor,
        identity_coordinates=identity_coordinates,
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
    """Validate scalar controls for Riemannian Armijo optimization."""
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
    """Map an optimizer result from whitened back to physical coordinates."""
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
    """Return coefficients and covariance for one microscopic VAR system."""
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
    """Run ComplexBox-style proxy pre-optimization on direct VAR coefficients."""
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
    """Run Riemannian proxy pre-optimization on direct VAR coefficients."""
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
    """Execute the SSDI proxy pre-optimization stage for one backend."""
    if isinstance(system, VARSystem):
        if optimizer == "adaptive":
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
    if optimizer == "adaptive":
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
    """Execute the full spectral-DD refinement stage for one backend."""
    opts = dict(options)
    if optimizer == "adaptive":
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
    optimizer: Optimizer = "adaptive",
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
    aliases = {"complexbox": "adaptive", "riemannian_armijo": "armijo"}
    optimizer = aliases.get(optimizer, optimizer)
    if optimizer not in ("adaptive", "armijo"):
        raise ValueError("optimizer must be 'adaptive' or 'armijo'")
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





@dataclass(frozen=True)
class DDOptimizationResult:
    """Backend-independent result of DD subspace optimization.

    Attributes
    ----------
    objective
        Sorted final objective values, shape ``(runs,)``.
    projection
        Corresponding row-orthonormal projections in physical observation
        coordinates, shape ``(runs, m, n)``.
    iterations
        Number/index of optimizer states using the backend's documented
        convention. Existing backend conventions are preserved for backward
        numerical parity.
    converged
        Boolean success indicator per run. ComplexBox codes 1--3 count as
        converged. Riemannian Armijo codes 1--2 count as converged, while code
        3 denotes line-search failure.
    convergence
        Raw backend termination code. This is retained because the detailed
        code meanings differ between optimizer families.
    step_size
        Final step size per run.
    objective_evaluations
        Scalar objective evaluations per run. For the ComplexBox backend this
        equals its recorded iteration/state count; for Armijo it additionally
        includes rejected line-search candidates.
    gradient_evaluations
        Gradient evaluations per run.
    backtracking_evaluations
        Rejected line-search candidate evaluations. Zero for ComplexBox.
    history
        Optional normalized history with shape ``(runs, states, 4)`` and
        columns ``(objective, step_size, gradient_norm,
        cumulative_objective_evaluations)``. Entries after stopping are NaN.
    optimizer
        Optimizer backend name.
    objective_name
        DD objective name: ``"proxy"`` or ``"spectral"``.
    """

    objective: torch.Tensor
    projection: torch.Tensor
    iterations: torch.Tensor
    converged: torch.Tensor
    convergence: torch.Tensor
    step_size: torch.Tensor
    objective_evaluations: torch.Tensor
    gradient_evaluations: torch.Tensor
    backtracking_evaluations: torch.Tensor
    history: torch.Tensor | None
    optimizer: DDOptimizer
    objective_name: DDObjective


@dataclass(frozen=True)
class DDSSDIOptimizationResult:
    """Result of the canonical staged SSDI optimization workflow.

    The final ``objective`` and ``projection`` properties refer to the full
    spectral-DD refinement. ``preoptimization`` stores the 100-restart proxy
    search, while clustering metadata records the exact SSDI ``Lcluster``
    reduction between stages.
    """

    preoptimization: DDOptimizationResult
    cluster_representative_indices: torch.Tensor
    cluster_sizes: torch.Tensor
    cluster_distances: torch.Tensor
    spectral: DDOptimizationResult
    frequencies: torch.Tensor

    @property
    def objective(self) -> torch.Tensor:
        """Sorted final spectral-DD objectives."""
        return self.spectral.objective

    @property
    def projection(self) -> torch.Tensor:
        """Final spectral-DD projections in physical observation coordinates."""
        return self.spectral.projection


def _observation_dimension_and_template(system: Model) -> tuple[int, torch.Tensor]:
    """Return observation dimension and a model tensor for dtype/device."""
    if isinstance(system, VARSystem):
        coefficients = torch.as_tensor(system.coefficients)
        return system.n_variables, coefficients
    iss = _as_innovations_state_space(system)
    observation = torch.as_tensor(iss.observation)
    return int(observation.shape[-2]), observation


def _random_initial_projections(
    system: Model,
    output_dimension: int,
    runs: int,
    *,
    random_seed: int | None,
) -> torch.Tensor:
    """Generate reproducible row-orthonormal SSDI restart subspaces."""
    n, template = _observation_dimension_and_template(system)
    if not 1 <= int(output_dimension) < n:
        raise ValueError("output_dimension must be between 1 and n-1")
    if runs < 1:
        raise ValueError("preoptimization_runs must be positive")
    generator = None
    if random_seed is not None:
        generator = torch.Generator(device=template.device).manual_seed(int(random_seed))
    raw = torch.randn(
        (runs, int(output_dimension), n),
        dtype=template.dtype,
        device=template.device,
        generator=generator,
    )
    return orthonormalise_projection(raw)


def _complexbox_history(
    result: DDGradientSearchResult,
) -> torch.Tensor | None:
    """Normalize ComplexBox's three-column history to the common contract."""
    if result.history is None:
        return None
    history = result.history
    counts = torch.arange(
        1,
        history.shape[1] + 1,
        dtype=history.dtype,
        device=history.device,
    ).expand(history.shape[0], -1)
    valid = torch.isfinite(history[..., 0])
    counts = torch.where(valid, counts, torch.full_like(counts, torch.nan))
    return torch.cat((history, counts.unsqueeze(-1)), dim=-1)


def _normalise_result(
    result: DDGradientSearchResult | DDRiemannianSearchResult,
    *,
    optimizer: DDOptimizer,
    objective_name: DDObjective,
) -> DDOptimizationResult:
    """Map backend-specific result metadata onto one public result contract."""
    if optimizer == "adaptive":
        evaluations = result.iterations.clone()
        converged = result.convergence != 0
        backtracking = torch.zeros_like(result.iterations)
        history = _complexbox_history(result)
        gradient_evaluations = result.iterations.clone()
    else:
        if not isinstance(result, DDRiemannianSearchResult):
            raise TypeError("riemannian_armijo returned an unexpected result type")
        evaluations = result.objective_evaluations
        gradient_evaluations = result.gradient_evaluations
        backtracking = result.backtracking_evaluations
        converged = (result.convergence == 1) | (result.convergence == 2)
        history = result.history

    return DDOptimizationResult(
        objective=result.objective,
        projection=result.projection,
        iterations=result.iterations,
        converged=converged,
        convergence=result.convergence,
        step_size=result.step_size,
        objective_evaluations=evaluations,
        gradient_evaluations=gradient_evaluations,
        backtracking_evaluations=backtracking,
        history=history,
        optimizer=optimizer,
        objective_name=objective_name,
    )


def _backend_kwargs(
    optimizer_options: Mapping[str, Any] | None,
    *,
    reserved: set[str],
) -> dict[str, Any]:
    """Copy method-specific options while protecting the common API contract."""
    options = {} if optimizer_options is None else dict(optimizer_options)
    overlap = reserved.intersection(options)
    if overlap:
        names = ", ".join(sorted(overlap))
        raise ValueError(
            f"optimizer_options must not override common argument(s): {names}"
        )
    return options


def optimise_dynamical_dependence(
    system: Model,
    initial_projection: torch.Tensor | None = None,
    *,
    objective: DDObjective | None = None,
    output_dimension: int | None = None,
    optimizer: DDOptimizer = "adaptive",
    lags: int | None = None,
    frequencies: torch.Tensor | None = None,
    max_iterations: int | None = None,
    history: bool = False,
    optimizer_options: Mapping[str, Any] | None = None,
    preoptimization_runs: int = 100,
    cluster_tolerance: float = 0.01,
    frequency_points: int = 513,
    preoptimization_max_iterations: int = 10_000,
    spectral_max_iterations: int = 10_000,
    random_seed: int | None = 0,
    preoptimizer_options: Mapping[str, Any] | None = None,
    spectral_optimizer_options: Mapping[str, Any] | None = None,
) -> DDOptimizationResult | DDSSDIOptimizationResult:
    """Optimize dynamical independence using SSDI by default.

    With ``objective=None`` (the default), this executes the canonical
    Barnett--Seth/SSDI workflow: many proxy-DD restarts, normalized Grassmann
    clustering with ``Lcluster`` semantics, then full spectral-DD refinement
    from one representative per cluster.  For a :class:`VARSystem`, the proxy
    stage uses the whitened VAR coefficient sequence directly, exactly as
    MATLAB SSDI ``cak2ddx`` does; the spectral stage uses the equivalent
    innovations representation and transfer function.

    Supplying ``objective="proxy"`` or ``objective="spectral"`` preserves
    the previous single-stage API for research, testing, and backwards
    compatibility.

    Parameters
    ----------
    system
        Microscopic VAR/state-space/innovations model.
    initial_projection
        Optional initial row projections.  Staged SSDI expects ``(runs,m,n)``.
        If omitted, ``output_dimension`` is required and 100 random restarts
        are generated by default. Single-stage calls still require this input.
    objective
        ``None`` for the canonical staged SSDI workflow (default), or
        ``"proxy"``/``"spectral"`` for the legacy single-stage objectives.
    output_dimension
        Macro dimension used when staged SSDI generates random restarts.
    optimizer
        ``"adaptive"`` is the reference/default step policy. In staged mode
        its scientifically recommended variant 1 is used unless explicitly
        overridden. ``"armijo"`` runs the same SSDI scientific
        pipeline with the native batched Riemannian optimizer.
    preoptimization_runs
        Number of random proxy-DD restarts when ``initial_projection`` is not
        supplied. Default 100, matching the validated SSDI workflow.
    cluster_tolerance
        Strict normalized Grassmann-distance threshold used by ``Lcluster``.
    frequency_points
        Number of one-sided frequencies over ``[0, 0.5]`` when no explicit
        ``frequencies`` grid is supplied.
    preoptimization_max_iterations, spectral_max_iterations
        Stage-specific iteration ceilings. Both default to 10,000.
    random_seed
        Seed for generated restart subspaces. ``None`` requests nondeterministic
        sampling; the default 0 makes the scientific workflow reproducible.
    preoptimizer_options, spectral_optimizer_options
        Stage-specific backend options for staged SSDI.

    Returns
    -------
    DDSSDIOptimizationResult or DDOptimizationResult
        Staged result by default; legacy common single-stage result when an
        explicit objective is requested.

    References
    ----------
    Barnett, L. and Seth, A. K. (2023), Phys. Rev. E 108, 014304.
    MATLAB SSDI: ``opt_gd_ddx_mruns`` -> ``Lcluster`` -> ``opt_gd_dds_mruns``.
    """
    aliases = {"complexbox": "adaptive", "riemannian_armijo": "armijo"}
    optimizer = aliases.get(optimizer, optimizer)
    if optimizer not in ("adaptive", "armijo"):
        raise ValueError("optimizer must be 'adaptive' or 'armijo'")

    if objective is None:
        if max_iterations is not None:
            raise ValueError(
                "max_iterations is a single-stage option; use the staged "
                "preoptimization_max_iterations/spectral_max_iterations arguments"
            )
        if optimizer_options is not None:
            raise ValueError(
                "optimizer_options is a single-stage option; use "
                "preoptimizer_options/spectral_optimizer_options in staged SSDI"
            )
        if initial_projection is None:
            if output_dimension is None:
                raise ValueError(
                    "output_dimension is required when staged SSDI generates restarts"
                )
            initial_projection = _random_initial_projections(
                system,
                output_dimension,
                preoptimization_runs,
                random_seed=random_seed,
            )
        elif output_dimension is not None and int(output_dimension) != int(
            torch.as_tensor(initial_projection).shape[-2]
        ):
            raise ValueError("output_dimension does not match initial_projection")

        pre_options = {} if preoptimizer_options is None else dict(preoptimizer_options)
        spectral_options = (
            {} if spectral_optimizer_options is None else dict(spectral_optimizer_options)
        )
        if optimizer == "adaptive":
            pre_options.setdefault("variant", 1)
            spectral_options.setdefault("variant", 1)
        raw = run_ssdi_workflow(
            system,
            torch.as_tensor(initial_projection),
            optimizer=optimizer,
            lags=lags,
            frequencies=frequencies,
            frequency_points=frequency_points,
            cluster_tolerance=cluster_tolerance,
            preoptimization_max_iterations=preoptimization_max_iterations,
            spectral_max_iterations=spectral_max_iterations,
            history=history,
            preoptimizer_options=pre_options,
            spectral_optimizer_options=spectral_options,
        )
        return DDSSDIOptimizationResult(
            preoptimization=_normalise_result(
                raw.preoptimization, optimizer=optimizer, objective_name="proxy"
            ),
            cluster_representative_indices=raw.cluster_representative_indices,
            cluster_sizes=raw.cluster_sizes,
            cluster_distances=raw.cluster_distances,
            spectral=_normalise_result(
                raw.spectral, optimizer=optimizer, objective_name="spectral"
            ),
            frequencies=raw.frequencies,
        )

    if initial_projection is None:
        raise ValueError("initial_projection is required for single-stage optimization")
    if objective not in ("proxy", "spectral"):
        raise ValueError("objective must be 'proxy' or 'spectral'")
    if optimizer not in ("adaptive", "armijo"):
        raise ValueError("optimizer must be 'adaptive' or 'armijo'")
    if objective == "proxy" and frequencies is not None:
        raise ValueError("frequencies is only valid for objective='spectral'")
    if objective == "spectral":
        if frequencies is None:
            raise ValueError("frequencies is required for objective='spectral'")
        if lags is not None:
            raise ValueError("lags is only valid for objective='proxy'")

    options = _backend_kwargs(
        optimizer_options,
        reserved={"lags", "frequencies", "max_iterations", "history"},
    )
    if max_iterations is not None:
        options["max_iterations"] = max_iterations
    options["history"] = history

    if objective == "proxy":
        if lags is not None:
            options["lags"] = lags
        if optimizer == "adaptive":
            raw = optimise_dynamical_dependence_proxy(
                system,
                initial_projection,
                **options,
            )
        else:
            raw = optimise_dynamical_dependence_proxy_riemannian(
                system,
                initial_projection,
                **options,
            )
    else:
        assert frequencies is not None
        if optimizer == "adaptive":
            raw = optimise_dynamical_dependence_spectral(
                system,
                initial_projection,
                frequencies,
                **options,
            )
        else:
            raw = optimise_dynamical_dependence_spectral_riemannian(
                system,
                initial_projection,
                frequencies,
                **options,
            )

    return _normalise_result(
        raw,
        optimizer=optimizer,
        objective_name=objective,
    )
