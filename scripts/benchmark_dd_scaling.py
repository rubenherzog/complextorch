"""Multidimensional CPU/GPU scaling benchmark for DD optimization.

The benchmark exercises the unified :func:`optimise_dynamical_dependence` API
and compares its recommended/default ``complexbox`` backend with the optional
``riemannian_armijo`` backend using identical batched restart tensors.

Sweeps
------
- restart count: 1, 8, 32, 128;
- proxy lags: 4, 16, 64, 256;
- spectral frequencies: 17, 65, 257;
- observation/state dimension: 4, 8, 16, 32.

For every point it reports final objective quality, exact DD from the canonical
generalized-DARE path, objective evaluations, iterations, elapsed time,
initialization sensitivity, convergence rate, and numerical stability.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch

from complextorch import (
    InnovationsStateSpace,
    dynamical_dependence,
    optimise_dynamical_dependence,
    orthonormalise_projection,
)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _system(n: int, dtype: torch.dtype, device: torch.device) -> InnovationsStateSpace:
    generator = torch.Generator(device="cpu").manual_seed(20260807)
    raw_a = torch.randn((n, n), generator=generator, dtype=dtype)
    radius = torch.max(torch.abs(torch.linalg.eigvals(raw_a))).real
    transition = (0.55 * raw_a / radius).to(device)
    observation = (
        torch.eye(n, dtype=dtype)
        + 0.12 * torch.randn((n, n), generator=generator, dtype=dtype)
    ).to(device)
    gain = (0.18 * torch.randn((n, n), generator=generator, dtype=dtype)).to(device)
    raw_v = torch.randn((n, n), generator=generator, dtype=dtype)
    covariance = (
        raw_v @ raw_v.T + 0.75 * torch.eye(n, dtype=dtype)
    ).to(device)
    return InnovationsStateSpace(transition, observation, gain, covariance)


def _initializations(
    runs: int,
    output_dimension: int,
    n: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(99173)
    raw = torch.randn(
        (runs, output_dimension, n), generator=generator, dtype=dtype
    ).to(device)
    return orthonormalise_projection(raw)


def _orthonormality_error(projection: torch.Tensor) -> float:
    identity = torch.eye(
        projection.shape[-2], dtype=projection.dtype, device=projection.device
    ).expand(projection.shape[0], -1, -1)
    error = torch.linalg.matrix_norm(
        projection @ projection.transpose(-1, -2) - identity,
        ord="fro",
        dim=(-2, -1),
    )
    return float(error.max().detach().cpu())


def _summary(
    *,
    sweep: str,
    scale: int,
    objective: str,
    optimizer: str,
    system: InnovationsStateSpace,
    result,
    elapsed: float,
) -> dict:
    exact_dd = dynamical_dependence(system, result.projection, base=math.e)
    if exact_dd.ndim == 0:
        exact_dd = exact_dd.unsqueeze(0)
    finite = torch.isfinite(result.objective) & torch.isfinite(exact_dd)
    finite = finite & torch.isfinite(result.projection).all(dim=(-2, -1))
    line_search_failures = (
        result.convergence == 3
        if optimizer == "riemannian_armijo"
        else torch.zeros_like(result.converged)
    )
    return {
        "sweep": sweep,
        "scale": int(scale),
        "objective": objective,
        "optimizer": optimizer,
        "runs": int(result.objective.shape[0]),
        "elapsed_seconds": elapsed,
        "seconds_per_run": elapsed / result.objective.shape[0],
        "final_objective_best": float(result.objective.min().detach().cpu()),
        "final_objective_mean": float(result.objective.mean().detach().cpu()),
        "final_objective_std": float(
            result.objective.std(unbiased=False).detach().cpu()
        ),
        "exact_dd_best": float(exact_dd.min().detach().cpu()),
        "exact_dd_mean": float(exact_dd.mean().detach().cpu()),
        "exact_dd_std": float(exact_dd.std(unbiased=False).detach().cpu()),
        "objective_evaluations_mean": float(
            result.objective_evaluations.to(torch.float64).mean().cpu()
        ),
        "iterations_mean": float(result.iterations.to(torch.float64).mean().cpu()),
        "convergence_rate": float(
            result.converged.to(torch.float64).mean().cpu()
        ),
        "line_search_failure_rate": float(
            line_search_failures.to(torch.float64).mean().cpu()
        ),
        "finite_rate": float(finite.to(torch.float64).mean().cpu()),
        "max_orthonormality_error": _orthonormality_error(result.projection),
    }


def _time_call(device: torch.device, function, *args, **kwargs):
    _synchronize(device)
    start = time.perf_counter()
    result = function(*args, **kwargs)
    _synchronize(device)
    return result, time.perf_counter() - start


def _run_point(
    *,
    sweep: str,
    scale: int,
    objective: str,
    n: int,
    runs: int,
    output_dimension: int,
    max_iterations: int,
    dtype: torch.dtype,
    device: torch.device,
    lags: int | None = None,
    frequencies: int | None = None,
) -> list[dict]:
    system = _system(n, dtype, device)
    initial = _initializations(runs, output_dimension, n, dtype, device)
    frequency_grid = None
    if frequencies is not None:
        frequency_grid = torch.linspace(
            0.0, 0.5, frequencies, dtype=dtype, device=device
        )

    rows = []
    for optimizer, optimizer_options in (
        (
            "complexbox",
            {"variant": 1, "initial_step_size": 1e-3},
        ),
        (
            "riemannian_armijo",
            {"initial_step_size": 1.0},
        ),
    ):
        result, elapsed = _time_call(
            device,
            optimise_dynamical_dependence,
            system,
            initial,
            objective=objective,
            optimizer=optimizer,
            lags=lags,
            frequencies=frequency_grid,
            max_iterations=max_iterations,
            optimizer_options=optimizer_options,
        )
        rows.append(
            _summary(
                sweep=sweep,
                scale=scale,
                objective=objective,
                optimizer=optimizer,
                system=system,
                result=result,
                elapsed=elapsed,
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--max-iterations", type=int, default=100)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    rows: list[dict] = []

    for runs in (1, 8, 32, 128):
        rows.extend(
            _run_point(
                sweep="restarts",
                scale=runs,
                objective="proxy",
                n=5,
                runs=runs,
                output_dimension=2,
                lags=5,
                max_iterations=args.max_iterations,
                dtype=dtype,
                device=device,
            )
        )

    for lags in (4, 16, 64, 256):
        rows.extend(
            _run_point(
                sweep="lags",
                scale=lags,
                objective="proxy",
                n=5,
                runs=8,
                output_dimension=2,
                lags=lags,
                max_iterations=args.max_iterations,
                dtype=dtype,
                device=device,
            )
        )

    for frequencies in (17, 65, 257):
        rows.extend(
            _run_point(
                sweep="frequencies",
                scale=frequencies,
                objective="spectral",
                n=5,
                runs=8,
                output_dimension=2,
                frequencies=frequencies,
                max_iterations=args.max_iterations,
                dtype=dtype,
                device=device,
            )
        )

    for n in (4, 8, 16, 32):
        rows.extend(
            _run_point(
                sweep="dimension",
                scale=n,
                objective="proxy",
                n=n,
                runs=8,
                output_dimension=2,
                lags=n,
                max_iterations=args.max_iterations,
                dtype=dtype,
                device=device,
            )
        )

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
