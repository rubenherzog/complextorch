#!/usr/bin/env python3
"""Validate VAR and state-space autocovariances against analytical references."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import complextorch as ct
from complexbox import mvgc

from common import (
    PASS_EXACT,
    Report,
    covariance_to_block_toeplitz,
    general_ssm_autocovariances_reference,
    general_ssm_fixture,
    general_ssm_state_covariance,
    innovations_autocovariances_reference,
    innovations_from_general_ssm,
    var_autocovariances_reference,
    var_fixtures,
)


def _yule_walker_residual(coefficients: np.ndarray, autocovariances: np.ndarray) -> float:
    order = coefficients.shape[0]

    def gamma(lag: int) -> np.ndarray:
        return autocovariances[lag] if lag >= 0 else autocovariances[-lag].T

    residual = 0.0
    for lag in range(1, autocovariances.shape[0]):
        predicted = sum(
            coefficients[index] @ gamma(lag - index - 1) for index in range(order)
        )
        residual = max(residual, float(np.max(np.abs(gamma(lag) - predicted))))
    return residual


def run(output_dir: Path) -> Report:
    report = Report("autocovariances")
    max_lag = 8

    for fixture in var_fixtures():
        reference = var_autocovariances_reference(fixture, max_lag)
        system = ct.build_var_system(
            torch.as_tensor(fixture.coefficients),
            torch.as_tensor(fixture.innovation_covariance),
        )
        ct_values = ct.model_autocovariances(system, max_lag)[0].detach().cpu().numpy()
        cb_values, cb_q = mvgc.var_to_autocov(
            fixture.complexbox_coefficients,
            fixture.innovation_covariance,
            qmax=-max_lag,
        )
        cb_values = np.transpose(cb_values, (2, 0, 1))

        report.compare(
            f"{fixture.name}_ct_vs_independent",
            ct_values,
            reference,
            atol=3e-10,
        )
        report.compare(
            f"{fixture.name}_complexbox_vs_independent",
            cb_values,
            reference,
            atol=3e-10,
        )
        report.compare(
            f"{fixture.name}_ct_vs_complexbox",
            ct_values,
            cb_values,
            atol=3e-10,
        )
        report.compare(
            f"{fixture.name}_complexbox_exact_lag_count",
            np.asarray([cb_q]),
            np.asarray([max_lag]),
            atol=0.0,
            pass_status=PASS_EXACT,
        )
        report.scalar_bound(
            f"{fixture.name}_yule_walker_residual",
            _yule_walker_residual(fixture.coefficients, reference),
            upper=5e-12,
            metric="max_abs_recursion_residual",
        )
        toeplitz = covariance_to_block_toeplitz(reference)
        report.scalar_bound(
            f"{fixture.name}_block_toeplitz_psd",
            max(0.0, -float(np.min(np.linalg.eigvalsh(toeplitz)))),
            upper=2e-10,
            metric="negative_min_eigenvalue",
        )

        general = system.to_state_space()
        general_values = ct.model_autocovariances(general, max_lag)[0].detach().cpu().numpy()
        report.compare(
            f"{fixture.name}_var_vs_companion_general_ssm",
            general_values,
            ct_values,
            atol=3e-10,
            notes="Equality is assessed through observable autocovariances, not internal state labels.",
        )

    fixture = general_ssm_fixture()
    state_covariance = general_ssm_state_covariance(fixture)
    reference = general_ssm_autocovariances_reference(fixture, max_lag)
    system = ct.StateSpaceModel(
        transition=torch.as_tensor(fixture.transition),
        observation=torch.as_tensor(fixture.observation),
        process_covariance=torch.as_tensor(fixture.process_covariance),
        observation_covariance=torch.as_tensor(fixture.observation_covariance),
        state_covariance=torch.as_tensor(state_covariance),
    )
    ct_values = ct.model_autocovariances(system, max_lag).detach().cpu().numpy()
    report.compare(
        "general_ssm_ct_vs_independent",
        ct_values,
        reference,
        atol=3e-10,
    )

    converted = ct.innovations_form(system)
    cb_values, cb_q = mvgc.ss_to_autocov(
        fixture.transition,
        fixture.observation,
        converted.gain.detach().cpu().numpy(),
        converted.covariance.detach().cpu().numpy(),
        qmax=-max_lag,
    )
    cb_values = np.transpose(cb_values, (2, 0, 1))
    innovations_reference = innovations_autocovariances_reference(
        innovations_from_general_ssm(fixture), max_lag
    )
    report.compare(
        "general_ssm_complexbox_innovations_vs_independent_general",
        cb_values,
        reference,
        atol=4e-10,
        notes="The general and innovations representations must induce the same observable process.",
    )
    report.compare(
        "general_ssm_complexbox_innovations_vs_independent_innovations",
        cb_values,
        innovations_reference,
        atol=4e-10,
    )
    report.compare(
        "general_ssm_ct_vs_complexbox_innovations",
        ct_values,
        cb_values,
        atol=4e-10,
    )
    report.compare(
        "general_ssm_complexbox_exact_lag_count",
        np.asarray([cb_q]),
        np.asarray([max_lag]),
        atol=0.0,
        pass_status=PASS_EXACT,
    )
    toeplitz = covariance_to_block_toeplitz(reference)
    report.scalar_bound(
        "general_ssm_block_toeplitz_psd",
        max(0.0, -float(np.min(np.linalg.eigvalsh(toeplitz)))),
        upper=2e-10,
        metric="negative_min_eigenvalue",
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
