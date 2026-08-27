"""Differential and optimization primitives for prescribed dynamical design.

The design layer is intentionally model-agnostic. A user supplies a batched
callable mapping continuous design parameters to a capability vector.
ComplexTorch then provides batched finite-difference Jacobians, local
neutral-space geometry, level-set correction, multistart equality-constrained
optimization, and Pareto dominance filtering. Scientific capability
definitions remain in their existing measure modules.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

BatchedCapabilityFunction = Callable[[torch.Tensor], torch.Tensor]
BatchedObjectiveFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
BatchedPenaltyFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
BatchedValidityFunction = Callable[[torch.Tensor], torch.Tensor]

__all__ = [
    "DesignOptimizationResult",
    "LevelSetProjectionResult",
    "capability_mobility",
    "finite_difference_jacobian",
    "jacobian_rank",
    "neutral_projector",
    "optimise_prescribed_capabilities",
    "pareto_nondominated",
    "project_to_capability_level_set",
]


@dataclass(frozen=True)
class LevelSetProjectionResult:
    """Result of projecting designs onto an equality-constrained level set."""

    parameters: torch.Tensor
    capabilities: torch.Tensor
    max_error: torch.Tensor
    converged: torch.Tensor
    iterations: torch.Tensor


@dataclass(frozen=True)
class DesignOptimizationResult:
    """Result of batched multistart prescribed-capability optimization."""

    parameters: torch.Tensor
    capabilities: torch.Tensor
    objective: torch.Tensor
    max_constraint_error: torch.Tensor
    converged: torch.Tensor
    history: torch.Tensor | None = None


def _as_batched_parameters(
    parameters: torch.Tensor, *, batched: bool
) -> tuple[torch.Tensor, bool]:
    """Normalize one design or a leading design batch without reshaping parameters."""
    value = torch.as_tensor(parameters)
    if not value.is_floating_point():
        raise TypeError("design parameters must use a floating-point dtype")
    if value.ndim == 0:
        raise ValueError("design parameters must have at least one dimension")
    if batched:
        if value.shape[0] == 0:
            raise ValueError("design batch must contain at least one item")
        return value, False
    return value.unsqueeze(0), True


def _validate_batched_output(
    output: torch.Tensor,
    batch: int,
    *,
    name: str,
    check_finite: bool = True,
) -> torch.Tensor:
    """Validate and normalize a capability-like output to ``(batch, outputs)``."""
    value = torch.as_tensor(output)
    if value.ndim == 1:
        value = value.unsqueeze(-1)
    if value.ndim != 2 or value.shape[0] != batch:
        raise ValueError(f"{name} must return shape (batch, n_outputs)")
    if check_finite and not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} returned non-finite values")
    return value


def _validate_validity(
    function: BatchedValidityFunction,
    parameters: torch.Tensor,
    *,
    name: str = "validity_function",
) -> torch.Tensor:
    """Return a validated boolean validity mask for a leading parameter batch."""
    valid = torch.as_tensor(function(parameters), device=parameters.device)
    if valid.ndim != 1 or valid.numel() != parameters.shape[0]:
        raise ValueError(f"{name} must return shape (batch,)")
    return valid.bool()


def finite_difference_jacobian(
    function: BatchedCapabilityFunction,
    parameters: torch.Tensor,
    *,
    step: float = 1e-5,
    batched: bool = False,
    chunk_size: int | None = 128,
    validity_function: BatchedValidityFunction | None = None,
) -> torch.Tensor:
    r"""Return a central finite-difference Jacobian with batched perturbations.

    ``function`` must preserve the leading batch dimension. If the design has
    ``p`` scalar parameters, plus/minus perturbations are evaluated in parameter
    chunks rather than one Python call per scalar parameter. This preserves
    vectorized model evaluation while bounding the memory required for large
    design spaces such as full network matrices.

    Parameters
    ----------
    function
        Callable from ``(batch, *parameter_shape)`` to
        ``(batch, n_capabilities)``.
    parameters
        One design when ``batched=False`` or a leading batch of independent
        designs when ``batched=True``.
    step
        Positive absolute central-difference step.
    batched
        Interpret the leading input dimension as an independent design batch.
    chunk_size
        Maximum number of scalar parameter directions evaluated together.
        ``None`` evaluates all directions in one batch. The default bounds
        memory while retaining batched execution.
    validity_function
        Optional predicate on a parameter batch. The finite-difference stencil
        must remain inside the valid design domain. Invalid perturbations are
        rejected before ``function`` is evaluated; reduce ``step`` or use a
        validity-preserving parameterization near a boundary.

    Returns
    -------
    torch.Tensor
        Shape ``(n_capabilities, p)`` for one design or
        ``(batch, n_capabilities, p)`` for a batch.
    """
    if step <= 0:
        raise ValueError("step must be positive")
    if chunk_size is not None and chunk_size < 1:
        raise ValueError("chunk_size must be positive or None")

    design, single = _as_batched_parameters(parameters, batched=batched)
    batch = design.shape[0]
    parameter_shape = design.shape[1:]
    n_parameters = int(design[0].numel())
    if n_parameters == 0:
        raise ValueError("design parameters must be non-empty")

    flat = design.reshape(batch, n_parameters)
    width = n_parameters if chunk_size is None else min(chunk_size, n_parameters)
    derivatives: list[torch.Tensor] = []
    n_outputs: int | None = None

    for start in range(0, n_parameters, width):
        stop = min(start + width, n_parameters)
        local_width = stop - start
        plus = flat[:, None, :].expand(batch, local_width, n_parameters).clone()
        minus = plus.clone()
        local_index = torch.arange(local_width, device=flat.device)
        parameter_index = torch.arange(start, stop, device=flat.device)
        plus[:, local_index, parameter_index] += float(step)
        minus[:, local_index, parameter_index] -= float(step)
        perturbations = torch.cat((plus, minus), dim=1).reshape(
            batch * 2 * local_width, *parameter_shape
        )

        if validity_function is not None:
            valid = _validate_validity(validity_function, perturbations)
            if not bool(valid.all().item()):
                raise ValueError(
                    "finite-difference perturbation leaves the valid design domain; "
                    "reduce step or use a validity-preserving parameterization"
                )

        values = _validate_batched_output(
            function(perturbations),
            batch * 2 * local_width,
            name="function",
        )
        if n_outputs is None:
            n_outputs = values.shape[-1]
        elif values.shape[-1] != n_outputs:
            raise ValueError("function output dimension changed across perturbation batches")
        values = values.reshape(batch, 2 * local_width, n_outputs)
        derivative = (
            values[:, :local_width] - values[:, local_width:]
        ) / (2.0 * float(step))
        derivatives.append(derivative)

    jacobian = torch.cat(derivatives, dim=1).transpose(-1, -2)
    return jacobian[0] if single else jacobian


def jacobian_rank(
    jacobian: torch.Tensor,
    *,
    rtol: float | None = None,
    atol: float = 0.0,
) -> torch.Tensor:
    """Return numerical matrix rank from singular values with batch support."""
    matrix = torch.as_tensor(jacobian)
    if matrix.ndim < 2:
        raise ValueError("jacobian must be a matrix or matrix batch")
    if atol < 0 or (rtol is not None and rtol < 0):
        raise ValueError("rtol and atol must be non-negative")
    singular = torch.linalg.svdvals(matrix)
    if rtol is None:
        rtol = max(matrix.shape[-2:]) * torch.finfo(singular.dtype).eps
    threshold = torch.maximum(
        torch.as_tensor(float(atol), dtype=singular.dtype, device=singular.device),
        float(rtol) * singular[..., :1],
    )
    return (singular > threshold).sum(-1)


def neutral_projector(
    jacobian: torch.Tensor,
    *,
    rtol: float | None = None,
    atol: float = 0.0,
) -> torch.Tensor:
    r"""Return the orthogonal projector onto ``Null(J)``.

    A projector has fixed shape ``(..., p, p)`` even when numerical rank varies
    across a batch, which makes it a convenient batched representation of local
    design degeneracy.
    """
    matrix = torch.as_tensor(jacobian)
    if matrix.ndim < 2:
        raise ValueError("jacobian must be a matrix or matrix batch")
    if atol < 0 or (rtol is not None and rtol < 0):
        raise ValueError("rtol and atol must be non-negative")
    _, singular, vh = torch.linalg.svd(matrix, full_matrices=True)
    if rtol is None:
        rtol = max(matrix.shape[-2:]) * torch.finfo(singular.dtype).eps
    threshold = torch.maximum(
        torch.as_tensor(float(atol), dtype=singular.dtype, device=singular.device),
        float(rtol) * singular[..., :1],
    )
    rank_mask = singular > threshold
    p = matrix.shape[-1]
    if vh.shape[-2] > singular.shape[-1]:
        padding = torch.zeros(
            (*rank_mask.shape[:-1], vh.shape[-2] - singular.shape[-1]),
            dtype=torch.bool,
            device=matrix.device,
        )
        row_mask = torch.cat((rank_mask, padding), dim=-1)
    else:
        row_mask = rank_mask[..., :p]
    null_mask = ~row_mask
    weighted = vh * null_mask[..., :, None].to(vh.dtype)
    return vh.transpose(-1, -2).conj() @ weighted


def capability_mobility(
    untargeted_jacobian: torch.Tensor,
    target_jacobian: torch.Tensor,
    *,
    rtol: float | None = None,
    atol: float = 0.0,
) -> torch.Tensor:
    r"""Return first-order untargeted mobility along target-neutral directions.

    If ``P_N`` projects onto ``Null(J_target)``, this returns

    .. math::

       J_{free} P_N.

    Its nonzero singular values are basis-independent measures of how strongly
    untargeted capabilities can move while the target panel remains fixed to
    first order.
    """
    free = torch.as_tensor(untargeted_jacobian)
    target = torch.as_tensor(target_jacobian)
    if free.shape[:-2] != target.shape[:-2] or free.shape[-1] != target.shape[-1]:
        raise ValueError(
            "target and untargeted Jacobians must share batch and parameter dimensions"
        )
    return free @ neutral_projector(target, rtol=rtol, atol=atol)


def project_to_capability_level_set(
    parameters: torch.Tensor,
    function: BatchedCapabilityFunction,
    target: torch.Tensor,
    *,
    step: float = 1e-5,
    damping: float = 1e-10,
    tolerance: float = 1e-9,
    max_iterations: int = 20,
    line_search_steps: int = 8,
    jacobian_chunk_size: int | None = 128,
    validity_function: BatchedValidityFunction | None = None,
    batched: bool = False,
) -> LevelSetProjectionResult:
    r"""Project designs to ``function(parameters) = target`` by damped Newton steps.

    Each correction is the minimum-norm row-space step

    .. math::

       \Delta\theta = J^\top(JJ^\top+\lambda I)^{-1}r,

    evaluated with batched linear solves. Backtracking candidates for all active
    designs are evaluated together. This is a local correction method; it does
    not claim global feasibility.

    ``validity_function`` is evaluated before candidate capabilities. Invalid
    backtracking candidates are replaced by the current valid design for the
    batched capability evaluation and are never accepted. The central
    finite-difference stencil itself must also remain inside the valid domain.
    """
    if damping < 0:
        raise ValueError("damping must be non-negative")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if max_iterations < 1 or line_search_steps < 1:
        raise ValueError("iteration counts must be positive")

    design, single = _as_batched_parameters(parameters, batched=batched)
    batch = design.shape[0]
    target_tensor = torch.as_tensor(target, dtype=design.dtype, device=design.device)
    if target_tensor.ndim == 1:
        target_tensor = target_tensor.unsqueeze(0).expand(batch, -1)
    if target_tensor.ndim != 2 or target_tensor.shape[0] != batch:
        raise ValueError(
            "target must have shape (n_capabilities,) or (batch,n_capabilities)"
        )

    current = design.detach().clone()
    if validity_function is not None:
        valid_initial = _validate_validity(validity_function, current)
        if not bool(valid_initial.all().item()):
            raise ValueError("initial parameters must satisfy validity_function")

    iterations = torch.zeros(batch, dtype=torch.int64, device=design.device)
    converged = torch.zeros(batch, dtype=torch.bool, device=design.device)
    capabilities = _validate_batched_output(function(current), batch, name="function")
    if capabilities.shape != target_tensor.shape:
        raise ValueError("target capability dimension does not match function output")

    parameter_shape = current.shape[1:]
    n_capabilities = capabilities.shape[-1]
    identity = torch.eye(
        n_capabilities, dtype=design.dtype, device=design.device
    ).expand(batch, -1, -1)
    alphas = 0.5 ** torch.arange(
        line_search_steps, dtype=design.dtype, device=design.device
    )

    for iteration in range(1, max_iterations + 1):
        residual = capabilities - target_tensor
        error = residual.abs().amax(-1)
        newly = (~converged) & (error <= tolerance)
        iterations = torch.where(
            newly, torch.full_like(iterations, iteration - 1), iterations
        )
        converged = converged | newly
        active = ~converged
        if not bool(torch.any(active).item()):
            break

        jacobian = finite_difference_jacobian(
            function,
            current,
            step=step,
            batched=True,
            chunk_size=jacobian_chunk_size,
            validity_function=validity_function,
        )
        gram = jacobian @ jacobian.transpose(-1, -2)
        solved = torch.linalg.solve(
            gram + float(damping) * identity, residual.unsqueeze(-1)
        ).squeeze(-1)
        delta = (jacobian.transpose(-1, -2) @ solved.unsqueeze(-1)).squeeze(-1)
        delta = delta.reshape(batch, *parameter_shape)

        candidates = current[:, None] - alphas.reshape(
            1, -1, *([1] * len(parameter_shape))
        ) * delta[:, None]
        flat_candidates = candidates.reshape(
            batch * line_search_steps, *parameter_shape
        )

        if validity_function is None:
            valid = torch.ones(
                batch * line_search_steps, dtype=torch.bool, device=design.device
            )
            evaluation_candidates = flat_candidates
        else:
            valid = _validate_validity(validity_function, flat_candidates)
            fallback = current[:, None].expand(
                batch, line_search_steps, *parameter_shape
            ).reshape(batch * line_search_steps, *parameter_shape)
            mask = valid.reshape(
                batch * line_search_steps, *([1] * len(parameter_shape))
            )
            evaluation_candidates = torch.where(mask, flat_candidates, fallback)

        candidate_values = _validate_batched_output(
            function(evaluation_candidates),
            batch * line_search_steps,
            name="function",
        ).reshape(batch, line_search_steps, n_capabilities)
        candidate_error = (candidate_values - target_tensor[:, None]).square().sum(-1)
        old_error = residual.square().sum(-1)
        acceptable = candidate_error < old_error[:, None]
        acceptable = acceptable & valid.reshape(batch, line_search_steps)

        any_acceptable = acceptable.any(-1) & active
        first_index = acceptable.to(torch.int64).argmax(-1)
        batch_index = torch.arange(batch, device=design.device)
        chosen = candidates[batch_index, first_index]
        chosen_values = candidate_values[batch_index, first_index]
        current = torch.where(
            any_acceptable.reshape(batch, *([1] * len(parameter_shape))),
            chosen,
            current,
        )
        capabilities = torch.where(
            any_acceptable[:, None], chosen_values, capabilities
        )
        stalled = active & ~any_acceptable
        iterations = torch.where(
            stalled, torch.full_like(iterations, iteration), iterations
        )
        converged = converged | stalled

    residual = capabilities - target_tensor
    max_error = residual.abs().amax(-1)
    success = max_error <= tolerance
    unfinished = iterations == 0
    iterations = torch.where(
        unfinished, torch.full_like(iterations, max_iterations), iterations
    )
    result = LevelSetProjectionResult(
        parameters=current,
        capabilities=capabilities,
        max_error=max_error,
        converged=success,
        iterations=iterations,
    )
    if not single:
        return result
    return LevelSetProjectionResult(
        parameters=result.parameters[0],
        capabilities=result.capabilities[0],
        max_error=result.max_error[0],
        converged=result.converged[0],
        iterations=result.iterations[0],
    )


def optimise_prescribed_capabilities(
    initial_parameters: torch.Tensor,
    capability_function: BatchedCapabilityFunction,
    target: torch.Tensor,
    *,
    objective_function: BatchedObjectiveFunction | None = None,
    penalty_function: BatchedPenaltyFunction | None = None,
    validity_function: BatchedValidityFunction | None = None,
    steps: int = 500,
    learning_rate: float = 1e-2,
    constraint_weight: float = 1e4,
    tolerance: float = 1e-7,
    project_final: bool = True,
    projection_step: float = 1e-5,
    projection_jacobian_chunk_size: int | None = 128,
    history: bool = False,
) -> DesignOptimizationResult:
    r"""Run batched multistart optimization for prescribed capabilities.

    The leading dimension of ``initial_parameters`` indexes independent design
    starts. All starts are optimized simultaneously with Adam. The objective
    and any additional soft penalty remain separate from the equality target.
    Optionally, the final designs are corrected to the target level set with
    :func:`project_to_capability_level_set`.

    If ``validity_function`` is supplied, every initial design must be valid.
    Adam proposals that leave the valid domain are reverted per run before the
    next capability evaluation. For hard-constrained problems, a smooth
    validity-preserving parameterization remains preferable because rejected
    Adam steps do not alter the optimizer's internal moment estimates.

    This routine is deliberately Euclidean and generic. Specialized manifold
    optimization of DD projections remains in :mod:`complextorch.dd_optimization`.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    if learning_rate <= 0 or constraint_weight <= 0 or tolerance <= 0:
        raise ValueError(
            "learning_rate, constraint_weight, and tolerance must be positive"
        )

    design = torch.as_tensor(initial_parameters)
    if design.ndim < 2 or design.shape[0] == 0:
        raise ValueError("initial_parameters must have shape (runs, *parameter_shape)")
    if not design.is_floating_point():
        raise TypeError("initial_parameters must use a floating-point dtype")
    if validity_function is not None:
        valid_initial = _validate_validity(validity_function, design)
        if not bool(valid_initial.all().item()):
            raise ValueError("initial_parameters must satisfy validity_function")

    target_tensor = torch.as_tensor(target, dtype=design.dtype, device=design.device)
    if target_tensor.ndim == 1:
        target_tensor = target_tensor.unsqueeze(0).expand(design.shape[0], -1)
    if target_tensor.ndim != 2 or target_tensor.shape[0] != design.shape[0]:
        raise ValueError(
            "target must have shape (n_capabilities,) or (runs,n_capabilities)"
        )

    parameter = design.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([parameter], lr=float(learning_rate))
    trace = None
    if history:
        trace = torch.empty(
            (steps, design.shape[0], 3), dtype=design.dtype, device=design.device
        )

    for iteration in range(steps):
        optimizer.zero_grad()
        capabilities = _validate_batched_output(
            capability_function(parameter),
            design.shape[0],
            name="capability_function",
            check_finite=False,
        )
        if capabilities.shape != target_tensor.shape:
            raise ValueError(
                "target capability dimension does not match capability_function"
            )
        residual = capabilities - target_tensor
        constraint = residual.square().sum(-1)

        if objective_function is None:
            objective = torch.zeros_like(constraint)
        else:
            objective = torch.as_tensor(objective_function(parameter, capabilities))
            if objective.shape != constraint.shape:
                raise ValueError("objective_function must return shape (runs,)")

        if penalty_function is None:
            penalty = torch.zeros_like(constraint)
        else:
            penalty = torch.as_tensor(penalty_function(parameter, capabilities))
            if penalty.shape != constraint.shape:
                raise ValueError("penalty_function must return shape (runs,)")

        loss_per_run = float(constraint_weight) * constraint + objective + penalty
        loss_per_run.mean().backward()
        previous = parameter.detach().clone()
        optimizer.step()

        if validity_function is not None:
            proposed_valid = _validate_validity(
                validity_function, parameter.detach()
            )
            with torch.no_grad():
                mask = proposed_valid.reshape(
                    design.shape[0], *([1] * (parameter.ndim - 1))
                )
                parameter.copy_(torch.where(mask, parameter, previous))

        if trace is not None:
            trace[iteration, :, 0] = constraint.detach().sqrt()
            trace[iteration, :, 1] = objective.detach()
            trace[iteration, :, 2] = penalty.detach()

    final = parameter.detach()
    if project_final:
        projected = project_to_capability_level_set(
            final,
            capability_function,
            target_tensor,
            step=projection_step,
            tolerance=tolerance,
            jacobian_chunk_size=projection_jacobian_chunk_size,
            validity_function=validity_function,
            batched=True,
        )
        final = projected.parameters
        capabilities = projected.capabilities
    else:
        capabilities = _validate_batched_output(
            capability_function(final),
            design.shape[0],
            name="capability_function",
        )

    max_error = (capabilities - target_tensor).abs().amax(-1)
    if objective_function is None:
        objective = torch.zeros_like(max_error)
    else:
        objective = torch.as_tensor(
            objective_function(final, capabilities)
        ).detach()
    return DesignOptimizationResult(
        parameters=final,
        capabilities=capabilities.detach(),
        objective=objective,
        max_constraint_error=max_error,
        converged=max_error <= tolerance,
        history=trace,
    )


def pareto_nondominated(
    objectives: torch.Tensor,
    *,
    maximize: torch.Tensor | list[bool] | tuple[bool, ...] | None = None,
    atol: float = 0.0,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Return a mask of Pareto-nondominated rows with explicit orientation.

    Parameters
    ----------
    objectives
        Tensor of shape ``(designs, objectives)``.
    maximize
        Boolean orientation for each objective. ``False`` means minimize;
        ``True`` means maximize. Defaults to minimizing every objective.
    atol
        Absolute comparison tolerance.
    chunk_size
        Number of candidate rows compared at once to bound ``O(N^2)`` memory.
    """
    values = torch.as_tensor(objectives)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("objectives must have shape (designs, objectives)")
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("objectives must contain only finite values")
    if atol < 0 or chunk_size < 1:
        raise ValueError("atol must be non-negative and chunk_size positive")

    if maximize is None:
        orientation = torch.zeros(
            values.shape[1], dtype=torch.bool, device=values.device
        )
    else:
        orientation = torch.as_tensor(
            maximize, dtype=torch.bool, device=values.device
        )
        if orientation.ndim != 1 or orientation.numel() != values.shape[1]:
            raise ValueError("maximize must contain one boolean per objective")

    normalized = torch.where(orientation, -values, values)
    nondominated = torch.ones(
        values.shape[0], dtype=torch.bool, device=values.device
    )
    tolerance = float(atol)
    for start in range(0, values.shape[0], chunk_size):
        stop = min(start + chunk_size, values.shape[0])
        candidate = normalized[start:stop]
        no_worse = normalized[:, None, :] <= candidate[None, :, :] + tolerance
        strictly_better = normalized[:, None, :] < candidate[None, :, :] - tolerance
        dominated = (no_worse.all(-1) & strictly_better.any(-1)).any(0)
        nondominated[start:stop] = ~dominated
    return nondominated
