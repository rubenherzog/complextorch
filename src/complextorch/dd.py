"""Public dynamical-independence optimisation API.

The default call executes the staged SSDI workflow inherited from Barnett--Seth
and ComplexBox: proxy-DD pre-optimisation, Grassmann clustering, then spectral
DD refinement. Explicit ``objective="proxy"`` or ``objective="spectral"``
requests retain the previous one-stage dispatcher for backward compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
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
from .dd_ssdi import optimise_ssdi_workflow

DDObjective = Literal["proxy", "spectral"]
DDOptimizer = Literal["complexbox", "riemannian_armijo"]
DDWorkflow = Literal["direct", "ssdi"]


@dataclass(frozen=True)
class DDOptimizationResult:
    """Backend-independent result of DD subspace optimisation.

    Attributes
    ----------
    objective
        Sorted final objective values, shape ``(runs,)``. In the default staged
        workflow these are the spectral-DD values of the refined cluster
        representatives.
    projection
        Corresponding row-orthonormal projections in physical observation
        coordinates, shape ``(runs, m, n)``.
    iterations
        Number/index of optimizer states using the backend's documented
        convention. Existing backend conventions are preserved for numerical
        parity.
    converged
        Boolean success indicator per returned run.
    convergence
        Raw backend termination code.
    step_size
        Final step size per returned run.
    objective_evaluations
        Scalar objective evaluations per returned run.
    gradient_evaluations
        Gradient evaluations per returned run.
    backtracking_evaluations
        Rejected line-search candidate evaluations. Zero for ComplexBox.
    history
        Optional normalized optimization history.
    optimizer
        Numerical optimiser backend.
    objective_name
        DD objective represented by the top-level endpoints.
    workflow
        ``"ssdi"`` for the default proxy→cluster→spectral workflow or
        ``"direct"`` for an explicitly selected one-stage objective.
    preoptimization
        Complete normalized proxy result for staged SSDI, otherwise ``None``.
    cluster_indices
        Indices into the sorted proxy result selected by SSDI ``Lcluster``.
    cluster_sizes
        Number of proxy endpoints represented by each selected cluster.
    cluster_distances
        Pairwise normalized maximum-principal-angle Grassmann distance matrix
        among all sorted proxy endpoints.
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
    workflow: DDWorkflow = "direct"
    preoptimization: DDOptimizationResult | None = None
    cluster_indices: torch.Tensor | None = None
    cluster_sizes: torch.Tensor | None = None
    cluster_distances: torch.Tensor | None = None


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


def _optimise_direct(
    system: Model,
    initial_projection: torch.Tensor,
    *,
    objective: DDObjective,
    optimizer: DDOptimizer,
    lags: int | None,
    frequencies: torch.Tensor | None,
    max_iterations: int | None,
    history: bool,
    optimizer_options: Mapping[str, Any] | None,
) -> DDOptimizationResult:
    """Preserve the established explicit one-stage optimizer behavior."""
    if objective not in ("proxy", "spectral"):
        raise ValueError("objective must be 'proxy' or 'spectral'")
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


def optimise_dynamical_dependence(
    system: Model,
    initial_projection: torch.Tensor | None = None,
    *,
    objective: DDObjective | None = None,
    optimizer: DDOptimizer = "complexbox",
    output_dimension: int | None = None,
    restarts: int = 100,
    lags: int | None = None,
    frequencies: torch.Tensor | None = None,
    frequency_bins: int = 513,
    cluster_tolerance: float = 1e-6,
    max_iterations: int | None = None,
    history: bool = False,
    optimizer_options: Mapping[str, Any] | None = None,
    preoptimization_options: Mapping[str, Any] | None = None,
    spectral_options: Mapping[str, Any] | None = None,
    seed: int | None = None,
) -> DDOptimizationResult:
    r"""Optimize a dynamically independent macro-subspace.

    By default, this runs the complete SSDI algorithmic workflow inherited from
    Barnett--Seth / MATLAB SSDI / ComplexBox: many proxy-DD pre-optimizations,
    Grassmann ``Lcluster`` reduction, and spectral-DD refinement from one
    representative per cluster. For :class:`~complextorch.VARSystem` input, the
    proxy stage uses the transformed VAR coefficient sequence directly, exactly
    as SSDI ``cak2ddx`` does; the spectral stage uses the equivalent innovations
    state-space transfer function.

    Explicitly passing ``objective="proxy"`` or ``objective="spectral"``
    retains the previous one-stage dispatcher and backend-specific numerical
    semantics.

    Parameters
    ----------
    system
        Microscopic VAR/state-space/innovations model.
    initial_projection
        Optional initial row projection, shape ``(m,n)``, or independent
        restart batch, shape ``(runs,m,n)``. If omitted in staged mode,
        ``output_dimension`` is required and ``restarts`` random orthonormal
        subspaces are generated.
    objective
        ``None`` (default) for the complete SSDI workflow. ``"proxy"`` or
        ``"spectral"`` explicitly selects the legacy one-stage optimizer.
    optimizer
        ``"complexbox"`` is the default/reference gradient-search backend.
        ``"riemannian_armijo"`` is an optional native-Torch optimizer that
        executes the same staged scientific workflow.
    output_dimension
        Macro dimension used when staged random initializations are generated.
    restarts
        Number of proxy pre-optimization restarts generated when
        ``initial_projection`` is omitted. Default 100.
    lags
        Optional proxy horizon for staged state-space input or explicit direct
        proxy optimization. Staged VAR input always uses all VAR coefficients.
    frequencies
        Optional one-sided normalized frequency grid. If omitted in staged mode,
        ``frequency_bins`` equally spaced points on ``[0, 0.5]`` are used.
    frequency_bins
        Default staged spectral grid size. Default 513.
    cluster_tolerance
        Normalized maximum-principal-angle threshold used by the exact greedy
        SSDI ``Lcluster`` semantics. Default ``1e-6``.
    max_iterations
        Iteration ceiling for each proxy and spectral optimization. Staged mode
        defaults to 10,000 for each stage.
    history
        Record normalized optimizer histories.
    optimizer_options
        Backend-specific options applied to both staged phases, or to the
        selected direct objective.
    preoptimization_options, spectral_options
        Backend-specific overrides for the staged proxy and spectral phases.
        ComplexBox-compatible staged defaults are variant 1, step sizes
        ``1.0``/``0.1``, ``gdls=2``, and tolerances ``1e-8``/``1e-10``.
    seed
        Optional Torch seed used only when staged initial restarts are generated.

    Returns
    -------
    DDOptimizationResult
        In staged mode, the top-level fields describe the spectrally refined
        cluster representatives. ``preoptimization`` and the cluster fields
        retain all proxy-stage diagnostics.

    Notes
    -----
    The staged ComplexBox-compatible workflow defaults to gradient-search
    variant 1 rather than variant 2. The reference variant-2 update moves the
    current subspace even when its best-so-far scalar objective is not updated,
    so the reported scalar can cease to correspond to the returned projection.
    Variant 2 remains available explicitly through ``optimizer_options`` for
    literal reference-behavior studies.

    References
    ----------
    - Barnett, L. and Seth, A. K. (2023), Physical Review E 108, 014304.
    - ``lcbarnett/ssdi`` commit ``b38ce65f9df18916da216848560c1789e456c04f``.
    - ``bmilinkovic/complexbox`` commit
      ``87b5e2cd9bba22ddd978bade6f614da7d6190db2``.
    """
    if optimizer not in ("complexbox", "riemannian_armijo"):
        raise ValueError(
            "optimizer must be 'complexbox' or 'riemannian_armijo'"
        )

    if objective is not None:
        if initial_projection is None:
            raise ValueError(
                "initial_projection is required for explicit one-stage objective optimization"
            )
        if preoptimization_options is not None or spectral_options is not None:
            raise ValueError(
                "preoptimization_options/spectral_options are only valid for the staged SSDI workflow"
            )
        return _optimise_direct(
            system,
            initial_projection,
            objective=objective,
            optimizer=optimizer,
            lags=lags,
            frequencies=frequencies,
            max_iterations=max_iterations,
            history=history,
            optimizer_options=optimizer_options,
        )

    staged = optimise_ssdi_workflow(
        system,
        initial_projection,
        optimizer=optimizer,
        output_dimension=output_dimension,
        restarts=restarts,
        lags=lags,
        frequencies=frequencies,
        frequency_bins=frequency_bins,
        cluster_tolerance=cluster_tolerance,
        max_iterations=max_iterations,
        history=history,
        optimizer_options=optimizer_options,
        preoptimization_options=preoptimization_options,
        spectral_options=spectral_options,
        seed=seed,
    )
    pre = _normalise_result(
        staged.preoptimization,
        optimizer=optimizer,
        objective_name="proxy",
    )
    final = _normalise_result(
        staged.spectral,
        optimizer=optimizer,
        objective_name="spectral",
    )
    return replace(
        final,
        workflow="ssdi",
        preoptimization=pre,
        cluster_indices=staged.cluster_indices,
        cluster_sizes=staged.cluster_sizes,
        cluster_distances=staged.cluster_distances,
    )


__all__ = [
    "DDObjective",
    "DDOptimizationResult",
    "DDOptimizer",
    "DDWorkflow",
    "optimise_dynamical_dependence",
]
