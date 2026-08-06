#!/usr/bin/env python3
"""Validate the fixed synthetic systems before testing toolbox transformations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import (
    PASS_EXACT,
    Report,
    companion_matrix_np,
    general_ssm_fixture,
    general_ssm_state_covariance,
    spectral_radius_np,
    var_autocovariances_reference,
    var_fixtures,
    var_stationary_state_covariance,
)


def _simulate_var(coefficients: np.ndarray, covariance: np.ndarray, samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order, n_variables, _ = coefficients.shape
    burnin = 2_000
    innovations = rng.multivariate_normal(np.zeros(n_variables), covariance, size=samples + burnin)
    data = np.zeros((samples + burnin, n_variables), dtype=np.float64)
    for time in range(order, samples + burnin):
        value = innovations[time].copy()
        for lag in range(1, order + 1):
            value += coefficients[lag - 1] @ data[time - lag]
        data[time] = value
    return data[burnin:]


def _sample_autocovariance(data: np.ndarray, lag: int) -> np.ndarray:
    centered = data - data.mean(axis=0, keepdims=True)
    if lag == 0:
        return centered.T @ centered / centered.shape[0]
    return centered[lag:].T @ centered[:-lag] / (centered.shape[0] - lag)


def run(output_dir: Path) -> Report:
    report = Report("synthetic_ground_truth")
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, object] = {}

    for fixture in var_fixtures():
        companion = companion_matrix_np(fixture.coefficients)
        radius = spectral_radius_np(companion)
        report.scalar_bound(
            f"{fixture.name}_stationarity",
            radius,
            upper=1.0 - 1e-12,
            metric="spectral_radius",
        )
        min_cov_eigenvalue = float(np.min(np.linalg.eigvalsh(fixture.innovation_covariance)))
        report.scalar_bound(
            f"{fixture.name}_innovation_covariance_spd",
            max(0.0, -min_cov_eigenvalue),
            upper=0.0,
            metric="negative_min_eigenvalue",
            pass_status=PASS_EXACT,
        )
        state_covariance = var_stationary_state_covariance(fixture)
        noise = np.zeros_like(companion)
        n_variables = fixture.coefficients.shape[1]
        noise[:n_variables, :n_variables] = fixture.innovation_covariance
        residual = state_covariance - companion @ state_covariance @ companion.T - noise
        report.scalar_bound(
            f"{fixture.name}_stationary_lyapunov_residual",
            np.max(np.abs(residual)),
            upper=1e-11,
            metric="max_abs_residual",
        )

        analytic = var_autocovariances_reference(fixture, 2)
        simulated = _simulate_var(fixture.coefficients, fixture.innovation_covariance, 150_000, seed=1729)
        sample = np.stack([_sample_autocovariance(simulated, lag) for lag in range(3)])
        report.compare(
            f"{fixture.name}_monte_carlo_autocovariance",
            sample,
            analytic,
            atol=3.5e-2,
            notes="Independent simulation sanity check; tolerance reflects finite-sample error.",
        )
        arrays[f"{fixture.name}_coefficients_lag_first"] = fixture.coefficients
        arrays[f"{fixture.name}_innovation_covariance"] = fixture.innovation_covariance
        arrays[f"{fixture.name}_companion"] = companion
        arrays[f"{fixture.name}_state_covariance"] = state_covariance
        arrays[f"{fixture.name}_autocovariances_lag0_to_2"] = analytic
        metadata[fixture.name] = {"spectral_radius": radius}

    ssm = general_ssm_fixture()
    radius = spectral_radius_np(ssm.transition)
    report.scalar_bound(
        "general_ssm_stationarity",
        radius,
        upper=1.0 - 1e-12,
        metric="spectral_radius",
    )
    for name, covariance in (
        ("process", ssm.process_covariance),
        ("observation", ssm.observation_covariance),
    ):
        minimum = float(np.min(np.linalg.eigvalsh(covariance)))
        report.scalar_bound(
            f"general_ssm_{name}_covariance_spd",
            max(0.0, -minimum),
            upper=0.0,
            metric="negative_min_eigenvalue",
            pass_status=PASS_EXACT,
        )
    observability = np.concatenate(
        [ssm.observation @ np.linalg.matrix_power(ssm.transition, power) for power in range(3)],
        axis=0,
    )
    report.compare(
        "general_ssm_observability_rank",
        np.asarray([np.linalg.matrix_rank(observability)]),
        np.asarray([ssm.transition.shape[0]]),
        atol=0.0,
        pass_status=PASS_EXACT,
    )
    state_covariance = general_ssm_state_covariance(ssm)
    residual = (
        state_covariance
        - ssm.transition @ state_covariance @ ssm.transition.T
        - ssm.process_covariance
    )
    report.scalar_bound(
        "general_ssm_stationary_lyapunov_residual",
        np.max(np.abs(residual)),
        upper=1e-11,
        metric="max_abs_residual",
    )
    arrays.update(
        {
            "general_ssm_transition": ssm.transition,
            "general_ssm_observation": ssm.observation,
            "general_ssm_process_covariance": ssm.process_covariance,
            "general_ssm_observation_covariance": ssm.observation_covariance,
            "general_ssm_state_covariance": state_covariance,
        }
    )
    metadata[ssm.name] = {
        "spectral_radius": radius,
        "observability_rank": int(np.linalg.matrix_rank(observability)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "synthetic_ground_truth.npz", **arrays)
    (output_dir / "synthetic_ground_truth_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    report.write(output_dir)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.output)
    report.raise_for_failures()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
