"""Unified public API for dynamical-dependence optimization.

This module provides a thin, extensible dispatcher over the audited DD
optimization backends. The ComplexBox-compatible optimizer remains the default
and recommended implementation. Alternative optimizers are opt-in and retain
separate internal implementations.

The dispatcher intentionally does not reimplement DD objectives, innovation
whitening, Grassmann retractions, or optimizer update rules. Those remain in
``dd_optimization`` and backend-specific modules so that adding a new optimizer
does not duplicate the scientific preprocessing contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import torch

from .dd_optimization import (
    DDGradientSearchResult,
    Model,
    optimise_dynamical_dependence_proxy,
    optimise_dynamical_dependence_spectral,
)
from .dd_riemannian import (
    DDRiemannianSearchResult,
    optimise_dynamical_dependence_proxy_riemannian,
    optimise_dynamical_dependence_spectral_riemannian,
)

DDObjective = Literal["proxy", "spectral"]
DDOptimizer = Literal["complexbox", "riemannian_armijo"]


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
    if optimizer == "complexbox":
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
    initial_projection: torch.Tensor,
    *,
    objective: DDObjective,
    optimizer: DDOptimizer = "complexbox",
    lags: int | None = None,
    frequencies: torch.Tensor | None = None,
    max_iterations: int | None = None,
    history: bool = False,
    optimizer_options: Mapping[str, Any] | None = None,
) -> DDOptimizationResult:
    """Optimize a dynamical-dependence macro-subspace with a selected backend.

    Parameters
    ----------
    system
        Microscopic VAR/state-space/innovations model accepted by the existing
        DD optimization backends.
    initial_projection
        Initial row projection with shape ``(m, n)`` or batched independent
        restarts with shape ``(runs, m, n)``.
    objective
        ``"proxy"`` for the finite-lag SSDI proxy objective or ``"spectral"``
        for the spectral DD objective.
    optimizer
        Optimization backend. ``"complexbox"`` is the default, recommended,
        and reference-compatible optimizer. ``"riemannian_armijo"`` is an
        optional Riemannian gradient-descent backend with Armijo backtracking.
    lags
        Number of proxy lags. Used only for ``objective="proxy"``.
    frequencies
        Frequency grid required for ``objective="spectral"``.
    max_iterations
        Optional common iteration limit. If omitted, each backend keeps its
        established default, preserving the legacy API behavior.
    history
        Record normalized optimization history.
    optimizer_options
        Backend-specific keyword options. For ComplexBox these currently
        include ``variant``, ``initial_step_size``, ``gdls``, and ``tol``. For
        Riemannian Armijo these include ``initial_step_size``,
        ``armijo_constant``, ``backtrack_factor``, ``max_backtracks``,
        ``min_step``, ``gradient_tolerance``, and ``objective_tolerance``.

    Returns
    -------
    DDOptimizationResult
        Common result contract independent of the selected optimizer backend.

    Notes
    -----
    This function is intentionally a dispatcher rather than a second
    implementation of the scientific pipeline. Legacy public functions remain
    available and retain their exact backend-specific return types.
    """
    if objective not in ("proxy", "spectral"):
        raise ValueError("objective must be 'proxy' or 'spectral'")
    if optimizer not in ("complexbox", "riemannian_armijo"):
        raise ValueError(
            "optimizer must be 'complexbox' or 'riemannian_armijo'"
        )
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
        if optimizer == "complexbox":
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
        if optimizer == "complexbox":
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


__all__ = [
    "DDObjective",
    "DDOptimizationResult",
    "DDOptimizer",
    "optimise_dynamical_dependence",
]
