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
from .dd_ssdi import run_ssdi_workflow
from .control import _as_innovations_state_space
from .representations import VARSystem

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
    from .dd_optimization import orthonormalise_projection
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
    initial_projection: torch.Tensor | None = None,
    *,
    objective: DDObjective | None = None,
    output_dimension: int | None = None,
    optimizer: DDOptimizer = "complexbox",
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
        ``"complexbox"`` is the reference/default optimizer. In staged mode
        its scientifically recommended variant 1 is used unless explicitly
        overridden. ``"riemannian_armijo"`` runs the same SSDI scientific
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
    if optimizer not in ("complexbox", "riemannian_armijo"):
        raise ValueError("optimizer must be 'complexbox' or 'riemannian_armijo'")

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
        if optimizer == "complexbox":
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
    "DDSSDIOptimizationResult",
    "DDOptimizer",
    "optimise_dynamical_dependence",
]
