#!/usr/bin/env python3
"""Validate environment capture and ComplexTorch/ComplexBox axis conventions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import complextorch as ct
from complexbox import mvgc

from common import PASS_EXACT, Report, environment_payload, var_fixtures


def run(output_dir: Path) -> Report:
    report = Report("environment_conventions")
    payload = environment_payload()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "environment.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    x = torch.arange(2 * 7 * 3, dtype=torch.float64).reshape(2, 7, 3)
    complexbox_x = ct.to_complexbox_timeseries(x, squeeze_single=False)
    report.compare(
        "timeseries_ct_to_cb_axis_permutation",
        complexbox_x.cpu().numpy(),
        x.permute(2, 1, 0).cpu().numpy(),
        atol=0.0,
        pass_status=PASS_EXACT,
        notes="ComplexTorch (batch,time,variables) -> ComplexBox (variables,time,trials).",
    )
    roundtrip_x = ct.from_complexbox_timeseries(complexbox_x)
    report.compare(
        "timeseries_adapter_roundtrip_batched",
        roundtrip_x.cpu().numpy(),
        x.cpu().numpy(),
        atol=0.0,
        pass_status=PASS_EXACT,
    )

    single = x[0]
    single_cb = ct.to_complexbox_timeseries(single)
    report.compare(
        "timeseries_adapter_single_shape",
        np.asarray(single_cb.shape),
        np.asarray((3, 7)),
        atol=0.0,
        pass_status=PASS_EXACT,
    )
    single_back = ct.from_complexbox_timeseries(single_cb)
    report.compare(
        "timeseries_adapter_roundtrip_single",
        single_back[0].cpu().numpy(),
        single.cpu().numpy(),
        atol=0.0,
        pass_status=PASS_EXACT,
    )

    fixture = var_fixtures()[1]
    coefficients = torch.as_tensor(fixture.coefficients)
    complexbox_a = ct.to_complexbox_var(coefficients)
    report.compare(
        "var_ct_to_cb_axis_permutation",
        complexbox_a.cpu().numpy(),
        fixture.complexbox_coefficients,
        atol=0.0,
        pass_status=PASS_EXACT,
        notes="ComplexTorch (lag,target,source) -> ComplexBox (target,source,lag).",
    )
    coefficients_back = ct.from_complexbox_var(complexbox_a)
    report.compare(
        "var_adapter_roundtrip",
        coefficients_back[0].cpu().numpy(),
        fixture.coefficients,
        atol=0.0,
        pass_status=PASS_EXACT,
    )

    system = ct.build_var_system(coefficients, torch.as_tensor(fixture.innovation_covariance))
    cb_transition, cb_observation, cb_gain, cb_covariance = mvgc.var_to_ss(
        fixture.complexbox_coefficients, fixture.innovation_covariance
    )
    ct_innovations = ct.var_to_innovations_state_space(system)
    report.compare(
        "companion_transition_convention",
        ct_innovations.transition[0].detach().cpu().numpy(),
        cb_transition,
        atol=0.0,
        pass_status=PASS_EXACT,
    )
    report.compare(
        "innovations_observation_convention",
        ct_innovations.observation[0].detach().cpu().numpy(),
        cb_observation,
        atol=0.0,
        pass_status=PASS_EXACT,
    )
    report.compare(
        "innovations_gain_convention",
        ct_innovations.gain[0].detach().cpu().numpy(),
        cb_gain,
        atol=0.0,
        pass_status=PASS_EXACT,
    )
    report.compare(
        "innovation_covariance_convention",
        ct_innovations.innovation_covariance[0].detach().cpu().numpy(),
        cb_covariance,
        atol=0.0,
        pass_status=PASS_EXACT,
    )

    report.compare(
        "adapter_preserves_float64",
        np.asarray([complexbox_x.dtype == torch.float64], dtype=int),
        np.asarray([1]),
        atol=0.0,
        pass_status=PASS_EXACT,
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
