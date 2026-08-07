#!/usr/bin/env python3
"""Validate deterministic VAR, state-space, and innovations-form conversions."""
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
    dare_reference,
    general_ssm_fixture,
    general_ssm_state_covariance,
    innovations_from_var,
    var_fixtures,
)


def run(output_dir: Path) -> Report:
    report = Report("var_ss_conversions")

    for fixture in var_fixtures():
        system = ct.build_var_system(
            torch.as_tensor(fixture.coefficients),
            torch.as_tensor(fixture.innovation_covariance),
        )
        ct_innovations = ct.var_to_innovations_state_space(system)
        reference = innovations_from_var(fixture)
        cb_transition, cb_observation, cb_gain, cb_covariance = mvgc.var_to_ss(
            fixture.complexbox_coefficients,
            fixture.innovation_covariance,
        )

        report.compare(
            f"{fixture.name}_transition_vs_independent",
            ct_innovations.transition[0].detach().cpu().numpy(),
            reference.transition,
            atol=0.0,
            pass_status=PASS_EXACT,
        )
        report.compare(
            f"{fixture.name}_observation_vs_independent",
            ct_innovations.observation[0].detach().cpu().numpy(),
            reference.observation,
            atol=0.0,
            pass_status=PASS_EXACT,
        )
        report.compare(
            f"{fixture.name}_gain_vs_independent",
            ct_innovations.gain[0].detach().cpu().numpy(),
            reference.gain,
            atol=0.0,
            pass_status=PASS_EXACT,
        )
        report.compare(
            f"{fixture.name}_innovation_covariance_vs_independent",
            ct_innovations.innovation_covariance[0].detach().cpu().numpy(),
            reference.innovation_covariance,
            atol=0.0,
            pass_status=PASS_EXACT,
        )
        report.compare(
            f"{fixture.name}_transition_ct_vs_complexbox",
            ct_innovations.transition[0].detach().cpu().numpy(),
            cb_transition,
            atol=0.0,
            pass_status=PASS_EXACT,
        )
        report.compare(
            f"{fixture.name}_observation_ct_vs_complexbox",
            ct_innovations.observation[0].detach().cpu().numpy(),
            cb_observation,
            atol=0.0,
            pass_status=PASS_EXACT,
        )
        report.compare(
            f"{fixture.name}_gain_ct_vs_complexbox",
            ct_innovations.gain[0].detach().cpu().numpy(),
            cb_gain,
            atol=0.0,
            pass_status=PASS_EXACT,
        )
        report.compare(
            f"{fixture.name}_covariance_ct_vs_complexbox",
            ct_innovations.innovation_covariance[0].detach().cpu().numpy(),
            cb_covariance,
            atol=0.0,
            pass_status=PASS_EXACT,
        )

        general = system.to_state_space()
        report.compare(
            f"{fixture.name}_general_ssm_observation_projection",
            general.observation[0, :, : fixture.coefficients.shape[1]].detach().cpu().numpy(),
            np.eye(fixture.coefficients.shape[1]),
            atol=0.0,
            pass_status=PASS_EXACT,
            notes="This representation observes the present companion state; it is not matrix-identical to innovations form.",
        )
        report.compare(
            f"{fixture.name}_general_ssm_zero_observation_noise",
            general.observation_covariance[0].detach().cpu().numpy(),
            np.zeros_like(fixture.innovation_covariance),
            atol=0.0,
            pass_status=PASS_EXACT,
        )

    fixture = general_ssm_fixture()
    state_covariance = general_ssm_state_covariance(fixture)
    system = ct.StateSpaceModel(
        transition=torch.as_tensor(fixture.transition),
        observation=torch.as_tensor(fixture.observation),
        process_covariance=torch.as_tensor(fixture.process_covariance),
        observation_covariance=torch.as_tensor(fixture.observation_covariance),
        state_covariance=torch.as_tensor(state_covariance),
    )
    converted = ct.innovations_form(system)
    reference_prediction, reference_gain, reference_innovation = dare_reference(fixture)
    cb_gain, cb_innovation, cb_rep, cb_prediction = mvgc.mdare(
        fixture.transition,
        fixture.observation,
        fixture.process_covariance,
        fixture.observation_covariance,
    )
    report.compare(
        "general_ssm_prediction_covariance_vs_independent",
        converted.prediction_covariance.detach().cpu().numpy(),
        reference_prediction,
        atol=2e-10,
    )
    report.compare(
        "general_ssm_gain_vs_independent",
        converted.gain.detach().cpu().numpy(),
        reference_gain,
        atol=2e-10,
    )
    report.compare(
        "general_ssm_innovation_covariance_vs_independent",
        converted.covariance.detach().cpu().numpy(),
        reference_innovation,
        atol=2e-10,
    )
    report.compare(
        "general_ssm_prediction_covariance_ct_vs_complexbox",
        converted.prediction_covariance.detach().cpu().numpy(),
        cb_prediction,
        atol=2e-10,
    )
    report.compare(
        "general_ssm_gain_ct_vs_complexbox",
        converted.gain.detach().cpu().numpy(),
        cb_gain,
        atol=2e-10,
    )
    report.compare(
        "general_ssm_innovation_covariance_ct_vs_complexbox",
        converted.covariance.detach().cpu().numpy(),
        cb_innovation,
        atol=2e-10,
    )
    report.scalar_bound(
        "general_ssm_complexbox_conversion_residual",
        cb_rep,
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
