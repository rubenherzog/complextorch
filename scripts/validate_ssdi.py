"""Reproducible SSDI / state-space dynamical-dependence validation study.

This executable is intentionally outside the library package. It builds deterministic
innovations-form state-space systems with several graph structures, evaluates both
ComplexTorch DD optimizers, optionally evaluates ComplexBox, and summarizes
wall time, numerical stability, objective quality, and local-minimum geometry.

References are pinned in ``reports/ssdi_testing_report_001.md``.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from complextorch import (
    InnovationsStateSpace,
    dynamical_dependence,
    optimise_dynamical_dependence,
    orthonormalise_projection,
)

COMPLEXBOX_COMMIT = "87b5e2cd9bba22ddd978bade6f614da7d6190db2"
MVGC2_COMMIT = "b22d3f0f061dcc40ba0e6cb31e636feb3183436d"


def tnet5_mask(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Return the exact MVGC2 ``demo/tnet5.m`` binary connectivity mask."""
    return torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
        ],
        dtype=dtype,
    )


def tnet9_mask(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Return the exact MVGC2 ``demo/tnet9.m`` binary connectivity mask."""
    return torch.tensor(
        [
            [1, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 0, 1],
        ],
        dtype=dtype,
    )


def erdos_renyi_mask(
    n: int, edge_probability: float, *, seed: int, dtype: torch.dtype
) -> torch.Tensor:
    """Generate a directed Erdos--Renyi mask with retained self-dynamics."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    mask = (
        torch.rand((n, n), generator=generator, dtype=dtype) < edge_probability
    ).to(dtype)
    mask.fill_diagonal_(1.0)
    return mask


def modular_mask(
    n_modules: int,
    module_size: int,
    *,
    within_probability: float,
    between_probability: float,
    seed: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Generate a directed stochastic-block mask with explicit modular structure."""
    n = n_modules * module_size
    generator = torch.Generator(device="cpu").manual_seed(seed)
    groups = torch.arange(n) // module_size
    same = groups[:, None] == groups[None, :]
    probabilities = torch.where(
        same,
        torch.full((n, n), within_probability, dtype=dtype),
        torch.full((n, n), between_probability, dtype=dtype),
    )
    mask = (torch.rand((n, n), generator=generator, dtype=dtype) < probabilities).to(
        dtype
    )
    mask.fill_diagonal_(1.0)
    return mask


def random_mask(n: int, *, density: float, seed: int, dtype: torch.dtype) -> torch.Tensor:
    """Generate an unconstrained directed random-network mask."""
    return erdos_renyi_mask(n, density, seed=seed, dtype=dtype)


def _spectral_radius(matrix: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.abs(torch.linalg.eigvals(matrix))).real


def innovations_system_from_mask(
    mask: torch.Tensor,
    *,
    seed: int,
    target_radius: float = 0.72,
    gain_scale: float = 0.16,
) -> InnovationsStateSpace:
    """Build a stable identity-observation innovations system from a graph mask.

    The directed mask defines allowed state-transition coefficients. The random
    transition is rescaled to an exact target spectral radius, so graph cases
    differ structurally without conflating comparisons with instability.
    """
    mask = torch.as_tensor(mask)
    n = mask.shape[0]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn((n, n), generator=generator, dtype=mask.dtype) * mask
    radius = _spectral_radius(raw)
    if float(radius) == 0.0:
        raise ValueError("mask generated a zero transition matrix")
    transition = target_radius * raw / radius
    observation = torch.eye(n, dtype=mask.dtype)
    gain = gain_scale * torch.eye(n, dtype=mask.dtype)
    covariance = torch.eye(n, dtype=mask.dtype)
    return InnovationsStateSpace(transition, observation, gain, covariance)


def planted_modular_system(
    n_modules: int = 3,
    *,
    within_coupling: float = 0.15,
    self_coupling: float = 0.45,
    gain_scale: float = 0.12,
    dtype: torch.dtype = torch.float64,
) -> tuple[InnovationsStateSpace, torch.Tensor]:
    """Return a symmetric paired-module system with a planted closed macrospace.

    Each two-variable module has transition block ``[[a,b],[b,a]]``. The
    normalized module averages are invariant eigenmodes. Their span is therefore
    a known macro-dynamical subspace and provides an analytic structural target
    for projection-angle tests.
    """
    n = 2 * n_modules
    transition = torch.zeros((n, n), dtype=dtype)
    planted = torch.zeros((n_modules, n), dtype=dtype)
    for module in range(n_modules):
        i = 2 * module
        transition[i : i + 2, i : i + 2] = torch.tensor(
            [[self_coupling, within_coupling], [within_coupling, self_coupling]],
            dtype=dtype,
        )
        planted[module, i : i + 2] = 1.0 / math.sqrt(2.0)
    system = InnovationsStateSpace(
        transition,
        torch.eye(n, dtype=dtype),
        gain_scale * torch.eye(n, dtype=dtype),
        torch.eye(n, dtype=dtype),
    )
    return system, planted


def case_library(dtype: torch.dtype = torch.float64) -> dict[str, InnovationsStateSpace]:
    """Return deterministic graph-structured state-space validation cases."""
    return {
        "tnet5": innovations_system_from_mask(tnet5_mask(dtype), seed=101),
        "tnet9": innovations_system_from_mask(tnet9_mask(dtype), seed=102),
        "erdos_renyi_sparse": innovations_system_from_mask(
            erdos_renyi_mask(10, 0.18, seed=103, dtype=dtype), seed=203
        ),
        "erdos_renyi_dense": innovations_system_from_mask(
            erdos_renyi_mask(10, 0.55, seed=104, dtype=dtype), seed=204
        ),
        "modular": innovations_system_from_mask(
            modular_mask(
                3,
                4,
                within_probability=0.75,
                between_probability=0.08,
                seed=105,
                dtype=dtype,
            ),
            seed=205,
        ),
        "random_network": innovations_system_from_mask(
            random_mask(12, density=0.35, seed=106, dtype=dtype), seed=206
        ),
    }


def random_initial_projections(
    runs: int,
    output_dimension: int,
    input_dimension: int,
    *,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Create deterministic row-orthonormal Grassmann restart representatives."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn(
        (runs, output_dimension, input_dimension), generator=generator, dtype=dtype
    ).to(device)
    return orthonormalise_projection(raw)


def principal_angles_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return principal angles between row subspaces, in radians.

    Orthonormal representatives are obtained by QR, then the singular values of
    ``Qa.T @ Qb`` are the cosines of the principal angles.
    """
    a = torch.as_tensor(a)
    b = torch.as_tensor(b, dtype=a.dtype, device=a.device)
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape:
        raise ValueError("a and b must have the same two-dimensional shape")
    qa, _ = torch.linalg.qr(a.transpose(-1, -2), mode="reduced")
    qb, _ = torch.linalg.qr(b.transpose(-1, -2), mode="reduced")
    cosines = torch.linalg.svdvals(qa.transpose(-1, -2) @ qb).clamp(0.0, 1.0)
    return torch.acos(cosines)


def subspace_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Grassmann geodesic distance: Euclidean norm of principal angles."""
    return float(torch.linalg.vector_norm(principal_angles_rows(a, b)).cpu())


def pairwise_subspace_distances(projections: torch.Tensor) -> torch.Tensor:
    """Return the symmetric matrix of Grassmann geodesic distances."""
    projections = torch.as_tensor(projections)
    runs = projections.shape[0]
    distances = torch.zeros((runs, runs), dtype=projections.dtype)
    for i in range(runs):
        for j in range(i + 1, runs):
            value = subspace_distance(projections[i], projections[j])
            distances[i, j] = value
            distances[j, i] = value
    return distances


def local_minimum_clusters(
    projections: torch.Tensor, *, angle_tolerance: float = 1e-3
) -> list[list[int]]:
    """Cluster optimizer endpoints by connected components in subspace distance."""
    distances = pairwise_subspace_distances(projections)
    remaining = set(range(projections.shape[0]))
    clusters: list[list[int]] = []
    while remaining:
        seed = min(remaining)
        component = {seed}
        frontier = [seed]
        while frontier:
            i = frontier.pop()
            neighbors = {
                j
                for j in remaining
                if float(distances[i, j]) <= angle_tolerance
            }
            new = neighbors - component
            component.update(new)
            frontier.extend(sorted(new))
        remaining -= component
        clusters.append(sorted(component))
    return clusters


def _orthonormality_error(projections: torch.Tensor) -> float:
    identity = torch.eye(
        projections.shape[-2], dtype=projections.dtype, device=projections.device
    )
    gram = projections @ projections.transpose(-1, -2)
    return float(
        torch.linalg.matrix_norm(gram - identity, ord="fro", dim=(-2, -1))
        .max()
        .cpu()
    )


@dataclass(frozen=True)
class OptimizerSummary:
    case: str
    toolbox: str
    optimizer: str
    runs: int
    elapsed_seconds: float
    best_objective: float
    mean_objective: float
    std_objective: float
    best_exact_dd: float
    finite_rate: float
    convergence_rate: float
    max_orthonormality_error: float
    local_minimum_clusters: int
    largest_cluster: int
    median_pairwise_subspace_distance: float
    planted_distance: float | None


def _torch_summary(
    case_name: str,
    system: InnovationsStateSpace,
    initial: torch.Tensor,
    optimizer: str,
    *,
    lags: int,
    max_iterations: int,
    planted: torch.Tensor | None = None,
) -> tuple[OptimizerSummary, Any]:
    start = time.perf_counter()
    result = optimise_dynamical_dependence(
        system,
        initial,
        objective="proxy",
        optimizer=optimizer,
        lags=lags,
        max_iterations=max_iterations,
        history=True,
        optimizer_options=(
            {"variant": 1, "initial_step_size": 1e-3}
            if optimizer == "complexbox"
            else {"initial_step_size": 1.0}
        ),
    )
    elapsed = time.perf_counter() - start
    exact = dynamical_dependence(system, result.projection, base=math.e)
    if exact.ndim == 0:
        exact = exact.unsqueeze(0)
    finite = torch.isfinite(result.objective) & torch.isfinite(exact)
    finite &= torch.isfinite(result.projection).all(dim=(-2, -1))
    clusters = local_minimum_clusters(result.projection.detach().cpu())
    distances = pairwise_subspace_distances(result.projection.detach().cpu())
    upper = distances[
        torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
    ]
    median_distance = float(torch.median(upper)) if upper.numel() else 0.0
    planted_distance = None
    if planted is not None:
        planted_distance = min(
            subspace_distance(row.cpu(), planted.cpu()) for row in result.projection
        )
    summary = OptimizerSummary(
        case=case_name,
        toolbox="ComplexTorch",
        optimizer=optimizer,
        runs=int(result.objective.shape[0]),
        elapsed_seconds=elapsed,
        best_objective=float(result.objective.min().detach().cpu()),
        mean_objective=float(result.objective.mean().detach().cpu()),
        std_objective=float(result.objective.std(unbiased=False).detach().cpu()),
        best_exact_dd=float(exact.min().detach().cpu()),
        finite_rate=float(finite.to(torch.float64).mean().cpu()),
        convergence_rate=float(result.converged.to(torch.float64).mean().cpu()),
        max_orthonormality_error=_orthonormality_error(result.projection),
        local_minimum_clusters=len(clusters),
        largest_cluster=max(map(len, clusters)),
        median_pairwise_subspace_distance=median_distance,
        planted_distance=planted_distance,
    )
    return summary, result


def _complexbox_available() -> bool:
    try:
        import complexbox  # noqa: F401
    except ImportError:
        return False
    return True


def _complexbox_summary(
    case_name: str,
    system: InnovationsStateSpace,
    initial: torch.Tensor,
    *,
    lags: int,
    max_iterations: int,
    planted: torch.Tensor | None = None,
) -> tuple[OptimizerSummary, dict[str, Any]]:
    """Run ComplexBox's NumPy SSDI optimizer from identical initial subspaces."""
    from complexbox.ssdi.dd import iss2cak, iss2dd
    from complexbox.ssdi.optimise import opt_gd_ddx_mruns

    a = system.transition.detach().cpu().numpy()
    c = system.observation.detach().cpu().numpy()
    k = system.gain.detach().cpu().numpy()
    cak = iss2cak(a, c, k, lags)
    l0 = initial.detach().cpu().numpy().transpose(2, 1, 0)
    start = time.perf_counter()
    dds, ls, convergence, histories = opt_gd_ddx_mruns(
        cak,
        l0,
        maxiters=max_iterations,
        variant=1,
        gdsig0=1e-3,
        history=True,
    )
    elapsed = time.perf_counter() - start
    rows = torch.as_tensor(
        ls.transpose(2, 1, 0), dtype=initial.dtype, device="cpu"
    )
    exact = np.array([iss2dd(ls[:, :, i], a, c, k) for i in range(ls.shape[2])])
    finite = np.isfinite(dds) & np.isfinite(exact) & np.isfinite(ls).all(axis=(0, 1))
    clusters = local_minimum_clusters(rows)
    distances = pairwise_subspace_distances(rows)
    upper = distances[
        torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
    ]
    planted_distance = None
    if planted is not None:
        planted_distance = min(subspace_distance(row, planted) for row in rows)
    summary = OptimizerSummary(
        case=case_name,
        toolbox="ComplexBox",
        optimizer="complexbox_numpy",
        runs=int(dds.size),
        elapsed_seconds=elapsed,
        best_objective=float(np.min(dds)),
        mean_objective=float(np.mean(dds)),
        std_objective=float(np.std(dds)),
        best_exact_dd=float(np.min(exact)),
        finite_rate=float(np.mean(finite)),
        convergence_rate=float(np.mean(np.asarray(convergence) != 0)),
        max_orthonormality_error=_orthonormality_error(rows),
        local_minimum_clusters=len(clusters),
        largest_cluster=max(map(len, clusters)),
        median_pairwise_subspace_distance=(
            float(torch.median(upper)) if upper.numel() else 0.0
        ),
        planted_distance=planted_distance,
    )
    payload = {
        "objective": np.asarray(dds),
        "projection": rows,
        "convergence": np.asarray(convergence),
        "histories": histories,
        "exact_dd": exact,
    }
    return summary, payload


def cross_toolbox_projection_match(
    torch_projection: torch.Tensor, complexbox_projection: torch.Tensor
) -> dict[str, float]:
    """Compare endpoint sets by nearest principal-angle distance in both directions."""
    a = torch_projection.detach().cpu()
    b = complexbox_projection.detach().cpu()
    distances = torch.empty((a.shape[0], b.shape[0]), dtype=a.dtype)
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            distances[i, j] = subspace_distance(a[i], b[j])
    return {
        "torch_to_complexbox_max_nearest": float(distances.min(dim=1).values.max()),
        "complexbox_to_torch_max_nearest": float(distances.min(dim=0).values.max()),
        "best_pair_distance": float(distances.min()),
    }


def run_study(
    *,
    runs: int,
    output_dimension: int,
    lags: int,
    max_iterations: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    """Execute the complete deterministic validation/benchmark matrix."""
    systems = case_library(dtype)
    planted_system, planted = planted_modular_system(dtype=dtype)
    systems = {"planted_modular": planted_system, **systems}
    summaries: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for index, (name, cpu_system) in enumerate(systems.items()):
        n = cpu_system.observation.shape[0]
        m = min(output_dimension, n - 1)
        if name == "planted_modular":
            m = planted.shape[0]
        system = InnovationsStateSpace(
            cpu_system.transition.to(device),
            cpu_system.observation.to(device),
            cpu_system.gain.to(device),
            cpu_system.innovation_covariance.to(device),
        )
        initial = random_initial_projections(
            runs,
            m,
            n,
            seed=9000 + index,
            dtype=dtype,
            device=device,
        )
        planted_case = planted if name == "planted_modular" else None
        torch_results: dict[str, Any] = {}
        for optimizer in ("complexbox", "riemannian_armijo"):
            summary, result = _torch_summary(
                name,
                system,
                initial,
                optimizer,
                lags=lags,
                max_iterations=max_iterations,
                planted=planted_case,
            )
            summaries.append(asdict(summary))
            torch_results[optimizer] = result

        if _complexbox_available() and device.type == "cpu":
            cb_summary, cb = _complexbox_summary(
                name,
                system,
                initial,
                lags=lags,
                max_iterations=max_iterations,
                planted=planted_case,
            )
            summaries.append(asdict(cb_summary))
            comparisons.append(
                {
                    "case": name,
                    "reference": f"ComplexBox@{COMPLEXBOX_COMMIT}",
                    "objective_max_abs_error": float(
                        np.max(
                            np.abs(
                                torch_results["complexbox"]
                                .objective.detach()
                                .cpu()
                                .numpy()
                                - cb["objective"]
                            )
                        )
                    ),
                    **cross_toolbox_projection_match(
                        torch_results["complexbox"].projection,
                        cb["projection"],
                    ),
                }
            )

    ground_truth_exact_dd = float(
        dynamical_dependence(planted_system, planted, base=math.e)
    )
    return {
        "metadata": {
            "complextorch_reference": "ssdi-testing",
            "complexbox_commit": COMPLEXBOX_COMMIT,
            "mvgc2_commit": MVGC2_COMMIT,
            "dtype": str(dtype),
            "device": str(device),
            "runs": runs,
            "output_dimension": output_dimension,
            "lags": lags,
            "max_iterations": max_iterations,
            "complexbox_available": _complexbox_available(),
        },
        "ground_truth": {
            "planted_modular_exact_dd": ground_truth_exact_dd,
            "planted_macro_projection": planted.tolist(),
        },
        "summaries": summaries,
        "cross_toolbox": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=32)
    parser.add_argument("--output-dimension", type=int, default=2)
    parser.add_argument("--lags", type=int, default=16)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be positive")
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    payload = run_study(
        runs=args.runs,
        output_dimension=args.output_dimension,
        lags=args.lags,
        max_iterations=args.max_iterations,
        dtype=dtype,
        device=device,
    )
    text = json.dumps(payload, indent=2)
    if args.output is None:
        print(text)
    else:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
