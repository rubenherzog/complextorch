"""Multidimensional CPU/GPU scaling benchmark for DD optimization.

This benchmark compares the frozen ComplexBox-compatible optimizer with the
Riemannian Armijo optimizer using the same *batched* restart tensor. It sweeps:

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
    optimise_dynamical_dependence_proxy,
    optimise_dynamical_dependence_proxy_riemannian,
    optimise_dynamical_dependence_spectral,
    optimise_dynamical_dependence_spectral_riemannian,
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
    riemannian: bool,
) -> dict:
    exact_dd = dynamical_dependence(system, result.projection, base=math.e)
    if exact_dd.ndim == 0:
        exact_dd = exact_dd.unsqueeze(0)
    finite = torch.isfinite(result.objective) & torch.isfinite(exact_dd)
    finite = finite & torch.isfinite(result.projection).all(dim=(-2, -1))
    if riemannian:
        objective_evaluations = result.objective_evaluations.to(torch.float64)
        convergence = (result.convergence == 1) | (result.convergence == 2)
        line_search_failures = result.convergence == 3
    else:
        objective_evaluations = result.iterations.to(torch.float64)
        convergence = result.convergence != 0
        line_search_failures = torch.zeros_like(convergence)
    iterations = result.iterations.to(torch.float64)
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
        "objective_evaluations_mean": float(objective_evaluations.mean().cpu()),
        "iterations_mean": float(iterations.mean().cpu()),
        "convergence_rate": float(convergence.to(torch.float64).mean().cpu()),
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


def _proxy_point(
    *,
    sweep: str,
    scale: int,
    n: int,
    runs: int,
    output_dimension: int,
    lags: int,
    max_iterations: int,
    dtype: torch.dtype,
    device: torch.device,
) -> list[dict]:
    system = _system(n, dtype, device)
    initial = _initializations(runs, output_dimension, n, dtype, device)
    baseline, baseline_time = _time_call(
        device,
        optimise_dynamical_dependence_proxy,
        system,
        initial,
        lags=lags,
        max_iterations=max_iterations,
        variant=1,
        initial_step_size=1e-3,
    )
    riemannian, riemannian_time = _time_call(
        device,
        optimise_dynamical_dependence_proxy_riemannian,
        system,
        initial,
        lags=lags,
        max_iterations=max_iterations,
        initial_step_size=1.0,
    )
    return [
        _summary(
            sweep=sweep,
            scale=scale,
            objective="proxy",
            optimizer="complexbox_baseline",
            system=system,
            result=baseline,
            elapsed=baseline_time,
            riemannian=False,
        ),
        _summary(
            sweep=sweep,
            scale=scale,
            objective="proxy",
            optimizer="riemannian_armijo_batched",
            system=system,
            result=riemannian,
            elapsed=riemannian_time,
            riemannian=True,
        ),
    ]


def _spectral_point(
    *,
    sweep: str,
    scale: int,
    n: int,
    runs: int,
    output_dimension: int,
    frequencies: int,
    max_iterations: int,
    dtype: torch.dtype,
    device: torch.device,
) -> list[dict]:
    system = _system(n, dtype, device)
    initial = _initializations(runs, output_dimension, n, dtype, device)
    frequency_grid = torch.linspace(
        0.0, 0.5, frequencies, dtype=dtype, device=device
    )
    baseline, baseline_time = _time_call(
        device,
        optimise_dynamical_dependence_spectral,
        system,
        initial,
        frequency_grid,
        max_iterations=max_iterations,
        variant=1,
        initial_step_size=1e-3,
    )
    riemannian, riemannian_time = _time_call(
        device,
        optimise_dynamical_dependence_spectral_riemannian,
        system,
        initial,
        frequency_grid,
        max_iterations=max_iterations,
        initial_step_size=1.0,
    )
    return [
        _summary(
            sweep=sweep,
            scale=scale,
            objective="spectral",
            optimizer="complexbox_baseline",
            system=system,
            result=baseline,
            elapsed=baseline_time,
            riemannian=False,
        ),
        _summary(
            sweep=sweep,
            scale=scale,
            objective="spectral",
            optimizer="riemannian_armijo_batched",
            system=system,
            result=riemannian,
            elapsed=riemannian_time,
            riemannian=True,
        ),
    ]


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
            _proxy_point(
                sweep="restarts",
                scale=runs,
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
            _proxy_point(
                sweep="lags",
                scale=lags,
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
            _spectral_point(
                sweep="frequencies",
                scale=frequencies,
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
            _proxy_point(
                sweep="dimension",
                scale=n,
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
