#!/usr/bin/env python3
"""Compare Lyapunov and DARE solutions against SciPy and ComplexBox."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov

import complextorch as ct
from complextorch.linalg import solve_discrete_lyapunov as ct_solve_discrete_lyapunov
from complexbox import mvgc

from common import PASS_EXACT, Report, companion_matrix_np, general_ssm_fixture, var_fixtures


def _riccati_residual(
    transition: np.ndarray,
    observation: np.ndarray,
    process_covariance: np.ndarray,
    observation_covariance: np.ndarray,
    prediction_covariance: np.ndarray,
    cross_covariance: np.ndarray | None = None,
) -> np.ndarray:
    cross = (
        np.zeros((transition.shape[0], observation.shape[0]), dtype=np.float64)
        if cross_covariance is None
        else cross_covariance
    )
    innovation = observation @ prediction_covariance @ observation.T + observation_covariance
    numerator = transition @ prediction_covariance @ observation.T + cross
    return (
        prediction_covariance
        - transition @ prediction_covariance @ transition.T
        - process_covariance
        + numerator @ np.linalg.solve(innovation, numerator.T)
    )


def run(output_dir: Path) -> Report:
    report = Report("dare_lyapunov")

    for fixture in var_fixtures():
        transition = companion_matrix_np(fixture.coefficients)
        n_variables = fixture.coefficients.shape[1]
        noise = np.zeros_like(transition)
        noise[:n_variables, :n_variables] = fixture.innovation_covariance
        scipy_solution = solve_discrete_lyapunov(transition, noise)

        ct_direct, direct_info = ct_solve_discrete_lyapunov(
            torch.as_tensor(transition), torch.as_tensor(noise), method="direct"
        )
        ct_doubling, doubling_info = ct_solve_discrete_lyapunov(
            torch.as_tensor(transition), torch.as_tensor(noise), method="doubling"
        )
        cb_schur = mvgc.dlyap(transition, noise)
        cb_smith, _ = mvgc.dlyap_aitr(transition, noise, maxrelerr=1e-11)

        report.compare(
            f"{fixture.name}_ct_direct_vs_scipy",
            ct_direct.detach().cpu().numpy(),
            scipy_solution,
            atol=2e-10,
        )
        report.compare(
            f"{fixture.name}_ct_doubling_vs_scipy",
            ct_doubling.detach().cpu().numpy(),
            scipy_solution,
            atol=2e-10,
        )
        report.compare(
            f"{fixture.name}_complexbox_schur_vs_scipy",
            cb_schur,
            scipy_solution,
            atol=2e-11,
        )
        report.compare(
            f"{fixture.name}_complexbox_smith_vs_scipy",
            cb_smith,
            scipy_solution,
            atol=2e-8,
            notes="Smith iteration is checked at its documented iterative tolerance.",
        )
        report.scalar_bound(
            f"{fixture.name}_ct_direct_reported_residual",
            direct_info.residual_max,
            upper=2e-10,
            metric="max_abs_lyapunov_residual",
        )
        report.scalar_bound(
            f"{fixture.name}_ct_doubling_reported_residual",
            doubling_info.residual_max,
            upper=2e-10,
            metric="max_abs_lyapunov_residual",
        )

    fixture = general_ssm_fixture()
    reference = solve_discrete_are(
        fixture.transition.T,
        fixture.observation.T,
        fixture.process_covariance,
        fixture.observation_covariance,
    )
    reference = 0.5 * (reference + reference.T)
    ct_prediction = ct.solve_dare(
        torch.as_tensor(fixture.transition),
        torch.as_tensor(fixture.observation),
        torch.as_tensor(fixture.process_covariance),
        torch.as_tensor(fixture.observation_covariance),
    )
    _, _, cb_rep, cb_prediction = mvgc.mdare(
        fixture.transition,
        fixture.observation,
        fixture.process_covariance,
        fixture.observation_covariance,
    )
    report.compare(
        "standard_dare_ct_vs_scipy",
        ct_prediction.detach().cpu().numpy(),
        reference,
        atol=2e-10,
    )
    report.compare(
        "standard_dare_complexbox_vs_scipy", cb_prediction, reference, atol=2e-10
    )
    report.compare(
        "standard_dare_ct_vs_complexbox",
        ct_prediction.detach().cpu().numpy(),
        cb_prediction,
        atol=2e-10,
    )
    report.scalar_bound(
        "standard_dare_ct_equation_residual",
        np.max(
            np.abs(
                _riccati_residual(
                    fixture.transition,
                    fixture.observation,
                    fixture.process_covariance,
                    fixture.observation_covariance,
                    ct_prediction.detach().cpu().numpy(),
                )
            )
        ),
        upper=2e-10,
        metric="max_abs_riccati_residual",
    )
    report.scalar_bound(
        "standard_dare_complexbox_reported_residual",
        cb_rep,
        upper=2e-10,
        metric="relative_riccati_residual",
    )

    cross_covariance = np.array(
        [[0.015, 0.000], [0.005, 0.008], [0.000, 0.010]], dtype=np.float64
    )
    joint_noise = np.block(
        [
            [fixture.process_covariance, cross_covariance],
            [cross_covariance.T, fixture.observation_covariance],
        ]
    )
    report.scalar_bound(
        "generalized_dare_joint_noise_spd",
        max(0.0, -float(np.min(np.linalg.eigvalsh(joint_noise)))),
        upper=0.0,
        metric="negative_min_eigenvalue",
        pass_status=PASS_EXACT,
    )
    generalized_reference = solve_discrete_are(
        fixture.transition.T,
        fixture.observation.T,
        fixture.process_covariance,
        fixture.observation_covariance,
        s=cross_covariance,
    )
    generalized_reference = 0.5 * (generalized_reference + generalized_reference.T)
    ct_generalized = ct.solve_generalized_dare(
        torch.as_tensor(fixture.transition),
        torch.as_tensor(fixture.observation),
        torch.as_tensor(fixture.process_covariance),
        torch.as_tensor(fixture.observation_covariance),
        torch.as_tensor(cross_covariance),
    )
    _, _, cb_generalized_rep, cb_generalized = mvgc.mdare(
        fixture.transition,
        fixture.observation,
        fixture.process_covariance,
        fixture.observation_covariance,
        cross_covariance,
    )
    report.compare(
        "generalized_dare_ct_vs_scipy",
        ct_generalized.detach().cpu().numpy(),
        generalized_reference,
        atol=5e-9,
        notes="ComplexTorch uses fixed-point Riccati iteration; SciPy/ComplexBox use QZ/Schur.",
    )
    report.compare(
        "generalized_dare_complexbox_vs_scipy",
        cb_generalized,
        generalized_reference,
        atol=2e-10,
    )
    report.scalar_bound(
        "generalized_dare_ct_equation_residual",
        np.max(
            np.abs(
                _riccati_residual(
                    fixture.transition,
                    fixture.observation,
                    fixture.process_covariance,
                    fixture.observation_covariance,
                    ct_generalized.detach().cpu().numpy(),
                    cross_covariance,
                )
            )
        ),
        upper=5e-9,
        metric="max_abs_riccati_residual",
    )
    report.scalar_bound(
        "generalized_dare_complexbox_reported_residual",
        cb_generalized_rep,
        upper=2e-10,
        metric="relative_riccati_residual",
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
