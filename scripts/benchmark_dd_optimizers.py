"""Benchmark frozen ComplexBox DD optimization against Riemannian Armijo.

The script reports, for proxy and/or spectral optimization:

1. final objective;
2. objective-evaluation count;
3. optimizer iterations;
4. elapsed CPU/GPU time;
5. sensitivity to initialization;
6. convergence rate;
7. numerical stability; and
8. exact dynamical dependence of the final physical subspace.

Examples
--------
python scripts/benchmark_dd_optimizers.py --device cpu --objective both
python scripts/benchmark_dd_optimizers.py --device cuda --objective proxy
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass

import torch

from complextorch import (
    InnovationsStateSpace,
    dynamical_dependence,
    optimise_dynamical_dependence_proxy,
    optimise_dynamical_dependence_proxy_riemannian,
    optimise_dynamical_dependence_spectral,
    optimise_dynamical_dependence_spectral_riemannian,
    orthonormalise_projection,
)


@dataclass(frozen=True)
class BenchmarkSummary:
    objective: str
    optimizer: str
    device: str
    dtype: str
    runs: int
    final_objective_best: float
    final_objective_mean: float
    final_objective_std: float
    objective_evaluations_mean: float
    iterations_mean: float
    elapsed_seconds: float
    seconds_per_run: float
    convergence_rate: float
    line_search_failure_rate: float
    finite_rate: float
    max_orthonormality_error: float
    exact_dd_best: float
    exact_dd_mean: float
    exact_dd_std: float


def _system(dtype: torch.dtype, device: torch.device) -> InnovationsStateSpace:
    """Stable nonidentity-innovation fixture shared by all benchmark methods."""
    generator = torch.Generator(device="cpu").manual_seed(20260807)
    n = 5
    raw_a = torch.randn((n, n), generator=generator, dtype=dtype)
    radius = torch.max(torch.abs(torch.linalg.eigvals(raw_a))).real
    a = (0.55 * raw_a / radius).to(device)
    c = (
        torch.eye(n, dtype=dtype)
        + 0.12 * torch.randn((n, n), generator=generator, dtype=dtype)
    ).to(device)
    k = (0.18 * torch.randn((n, n), generator=generator, dtype=dtype)).to(device)
    raw_v = torch.randn((n, n), generator=generator, dtype=dtype)
    v = (raw_v @ raw_v.T + 0.75 * torch.eye(n, dtype=dtype)).to(device)
    return InnovationsStateSpace(a, c, k, v)


def _initializations(
    *,
    runs: int,
    output_dimension: int,
    n_observations: int,
    dtype: torch.dtype,
    device: torch.device,
) -> list[torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(99173)
    values = []
    for _ in range(runs):
        raw = torch.randn(
            (output_dimension, n_observations),
            generator=generator,
            dtype=dtype,
        ).to(device)
        values.append(orthonormalise_projection(raw))
    return values


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _orthonormality_error(projection: torch.Tensor) -> float:
    identity = torch.eye(
        projection.shape[-2], dtype=projection.dtype, device=projection.device
    )
    error = torch.linalg.matrix_norm(
        projection @ projection.transpose(-1, -2) - identity, ord="fro"
    )
    return float(error.detach().cpu())


def _summarize(
    *,
    objective_name: str,
    optimizer_name: str,
    system: InnovationsStateSpace,
    outputs: list,
    elapsed_seconds: float,
    riemannian: bool,
) -> BenchmarkSummary:
    objectives = torch.stack([result.objective[0].detach() for result in outputs])
    projections = [result.projection[0] for result in outputs]
    exact_dd = torch.stack(
        [dynamical_dependence(system, matrix, base=math.e).detach() for matrix in projections]
    )
    finite = torch.tensor(
        [
            bool(torch.isfinite(result.objective).all())
            and bool(torch.isfinite(result.projection).all())
            for result in outputs
        ],
        dtype=torch.float64,
    )
    orth_errors = [_orthonormality_error(matrix) for matrix in projections]

    if riemannian:
        objective_evaluations = torch.tensor(
            [int(result.objective_evaluations[0]) for result in outputs],
            dtype=torch.float64,
        )
        iterations = torch.tensor(
            [int(result.iterations[0]) for result in outputs], dtype=torch.float64
        )
        # Codes 1/2 are successful convergence; code 3 is line-search failure.
        convergence = torch.tensor(
            [int(result.convergence[0]) in (1, 2) for result in outputs],
            dtype=torch.float64,
        )
        line_search_failure = torch.tensor(
            [int(result.convergence[0]) == 3 for result in outputs],
            dtype=torch.float64,
        )
    else:
        # The frozen ComplexBox baseline evaluates one objective candidate per
        # state after its initial objective, so scalar evaluations equal its
        # reported state-count convention for a single run.
        iterations = torch.tensor(
            [int(result.iterations[0]) for result in outputs], dtype=torch.float64
        )
        objective_evaluations = iterations.clone()
        convergence = torch.tensor(
            [int(result.convergence[0]) != 0 for result in outputs],
            dtype=torch.float64,
        )
        line_search_failure = torch.zeros_like(convergence)

    return BenchmarkSummary(
        objective=objective_name,
        optimizer=optimizer_name,
        device=str(system.transition.device),
        dtype=str(system.transition.dtype).replace("torch.", ""),
        runs=len(outputs),
        final_objective_best=float(objectives.min().cpu()),
        final_objective_mean=float(objectives.mean().cpu()),
        final_objective_std=float(objectives.std(unbiased=False).cpu()),
        objective_evaluations_mean=float(objective_evaluations.mean()),
        iterations_mean=float(iterations.mean()),
        elapsed_seconds=elapsed_seconds,
        seconds_per_run=elapsed_seconds / len(outputs),
        convergence_rate=float(convergence.mean()),
        line_search_failure_rate=float(line_search_failure.mean()),
        finite_rate=float(finite.mean()),
        max_orthonormality_error=max(orth_errors),
        exact_dd_best=float(exact_dd.min().cpu()),
        exact_dd_mean=float(exact_dd.mean().cpu()),
        exact_dd_std=float(exact_dd.std(unbiased=False).cpu()),
    )


def _run_proxy(system, initializations, max_iterations):
    baseline = []
    _synchronize(system.transition.device)
    start = time.perf_counter()
    for initial in initializations:
        baseline.append(
            optimise_dynamical_dependence_proxy(
                system,
                initial,
                max_iterations=max_iterations,
                variant=1,
                initial_step_size=1e-3,
            )
        )
    _synchronize(system.transition.device)
    baseline_time = time.perf_counter() - start

    riemannian = []
    _synchronize(system.transition.device)
    start = time.perf_counter()
    for initial in initializations:
        riemannian.append(
            optimise_dynamical_dependence_proxy_riemannian(
                system,
                initial,
                max_iterations=max_iterations,
                initial_step_size=1.0,
            )
        )
    _synchronize(system.transition.device)
    riemannian_time = time.perf_counter() - start
    return baseline, baseline_time, riemannian, riemannian_time


def _run_spectral(system, initializations, max_iterations, frequencies):
    baseline = []
    _synchronize(system.transition.device)
    start = time.perf_counter()
    for initial in initializations:
        baseline.append(
            optimise_dynamical_dependence_spectral(
                system,
                initial,
                frequencies,
                max_iterations=max_iterations,
                variant=1,
                initial_step_size=1e-3,
            )
        )
    _synchronize(system.transition.device)
    baseline_time = time.perf_counter() - start

    riemannian = []
    _synchronize(system.transition.device)
    start = time.perf_counter()
    for initial in initializations:
        riemannian.append(
            optimise_dynamical_dependence_spectral_riemannian(
                system,
                initial,
                frequencies,
                max_iterations=max_iterations,
                initial_step_size=1.0,
            )
        )
    _synchronize(system.transition.device)
    riemannian_time = time.perf_counter() - start
    return baseline, baseline_time, riemannian, riemannian_time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--objective", choices=("proxy", "spectral", "both"), default="both")
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--output-dimension", type=int, default=2)
    parser.add_argument("--frequencies", type=int, default=65)
    args = parser.parse_args()

    if args.runs < 2:
        raise ValueError("runs must be at least 2 to estimate initialization sensitivity")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    system = _system(dtype, device)
    n = system.observation.shape[-2]
    if not 1 <= args.output_dimension < n:
        raise ValueError("output-dimension must satisfy 1 <= m < n")
    initializations = _initializations(
        runs=args.runs,
        output_dimension=args.output_dimension,
        n_observations=n,
        dtype=dtype,
        device=device,
    )

    summaries = []
    if args.objective in ("proxy", "both"):
        baseline, bt, riemannian, rt = _run_proxy(
            system, initializations, args.max_iterations
        )
        summaries.append(
            _summarize(
                objective_name="proxy",
                optimizer_name="complexbox_baseline",
                system=system,
                outputs=baseline,
                elapsed_seconds=bt,
                riemannian=False,
            )
        )
        summaries.append(
            _summarize(
                objective_name="proxy",
                optimizer_name="riemannian_armijo",
                system=system,
                outputs=riemannian,
                elapsed_seconds=rt,
                riemannian=True,
            )
        )

    if args.objective in ("spectral", "both"):
        frequencies = torch.linspace(
            0.0, 0.5, args.frequencies, dtype=dtype, device=device
        )
        baseline, bt, riemannian, rt = _run_spectral(
            system, initializations, args.max_iterations, frequencies
        )
        summaries.append(
            _summarize(
                objective_name="spectral",
                optimizer_name="complexbox_baseline",
                system=system,
                outputs=baseline,
                elapsed_seconds=bt,
                riemannian=False,
            )
        )
        summaries.append(
            _summarize(
                objective_name="spectral",
                optimizer_name="riemannian_armijo",
                system=system,
                outputs=riemannian,
                elapsed_seconds=rt,
                riemannian=True,
            )
        )

    print(json.dumps([asdict(summary) for summary in summaries], indent=2))


if __name__ == "__main__":
    main()
