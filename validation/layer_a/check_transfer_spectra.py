#!/usr/bin/env python3
"""Validate transfer functions and spectra before spectral MVGC is tested."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import complextorch as ct
from complexbox import mvgc

from common import (
    Report,
    general_ssm_autocovariances_reference,
    general_ssm_fixture,
    general_ssm_state_covariance,
    innovations_from_general_ssm,
    innovations_transfer_reference,
    spectrum_from_transfer,
    var_autocovariances_reference,
    var_fixtures,
    var_transfer_reference,
)


def _spectral_diagnostics(report: Report, prefix: str, spectrum: np.ndarray) -> None:
    hermitian_error = max(
        float(np.max(np.abs(matrix - matrix.conj().T))) for matrix in spectrum
    )
    minimum_eigenvalue = min(
        float(np.min(np.linalg.eigvalsh(0.5 * (matrix + matrix.conj().T))))
        for matrix in spectrum
    )
    report.scalar_bound(
        f"{prefix}_hermitian",
        hermitian_error,
        upper=3e-11,
        metric="max_abs_hermitian_residual",
    )
    report.scalar_bound(
        f"{prefix}_positive_semidefinite",
        max(0.0, -minimum_eigenvalue),
        upper=3e-10,
        metric="negative_min_eigenvalue",
    )


def _integrated_gamma0(spectrum: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    omega = 2.0 * np.pi * frequencies
    return np.trapezoid(np.real(spectrum), omega, axis=0) / np.pi


def run(output_dir: Path) -> Report:
    report = Report("transfer_spectra")
    fres = 512
    frequencies = np.linspace(0.0, 0.5, fres + 1, dtype=np.float64)
    frequencies_t = torch.as_tensor(frequencies)

    for fixture in var_fixtures():
        reference = var_transfer_reference(fixture, frequencies)
        system = ct.build_var_system(
            torch.as_tensor(fixture.coefficients),
            torch.as_tensor(fixture.innovation_covariance),
        )
        innovations = ct.var_to_innovations_state_space(system)
        ct_transfer = ct.innovations_transfer_function(innovations, frequencies_t)[0]
        ct_transfer = ct_transfer.detach().cpu().numpy()
        cb_transfer = np.transpose(
            mvgc.var2trfun(fixture.complexbox_coefficients, fres), (2, 0, 1)
        )
        cb_ss = mvgc.var_to_ss(
            fixture.complexbox_coefficients, fixture.innovation_covariance
        )
        cb_ss_transfer = np.transpose(
            mvgc.ss2trfun(cb_ss[0], cb_ss[1], cb_ss[2], fres), (2, 0, 1)
        )

        report.compare(
            f"{fixture.name}_ct_transfer_vs_independent",
            ct_transfer,
            reference,
            atol=4e-10,
        )
        report.compare(
            f"{fixture.name}_complexbox_var_transfer_vs_independent",
            cb_transfer,
            reference,
            atol=4e-10,
        )
        report.compare(
            f"{fixture.name}_complexbox_var_vs_ss_transfer",
            cb_transfer,
            cb_ss_transfer,
            atol=4e-10,
        )
        report.compare(
            f"{fixture.name}_ct_vs_complexbox_transfer",
            ct_transfer,
            cb_transfer,
            atol=4e-10,
        )

        reference_spectrum = spectrum_from_transfer(
            reference, fixture.innovation_covariance
        )
        ct_spectrum = spectrum_from_transfer(
            ct_transfer, fixture.innovation_covariance
        )
        cb_spectrum = np.transpose(
            mvgc.var_to_cpsd(
                fixture.complexbox_coefficients,
                fixture.innovation_covariance,
                fres,
            ),
            (2, 0, 1),
        )
        report.compare(
            f"{fixture.name}_ct_spectrum_vs_independent",
            ct_spectrum,
            reference_spectrum,
            atol=8e-10,
        )
        report.compare(
            f"{fixture.name}_complexbox_spectrum_vs_independent",
            cb_spectrum,
            reference_spectrum,
            atol=8e-10,
        )
        _spectral_diagnostics(report, fixture.name, reference_spectrum)
        gamma0 = var_autocovariances_reference(fixture, 0)[0]
        report.compare(
            f"{fixture.name}_wiener_khinchin_gamma0",
            _integrated_gamma0(reference_spectrum, frequencies),
            gamma0,
            atol=2e-9,
        )

    fixture = general_ssm_fixture()
    reference_fixture = innovations_from_general_ssm(fixture)
    reference = innovations_transfer_reference(reference_fixture, frequencies)
    state_covariance = torch.as_tensor(general_ssm_state_covariance(fixture))
    general = ct.StateSpaceModel(
        torch.as_tensor(fixture.transition),
        torch.as_tensor(fixture.observation),
        torch.as_tensor(fixture.process_covariance),
        torch.as_tensor(fixture.observation_covariance),
        state_covariance,
    )
    converted = ct.innovations_form(general)
    innovations = ct.InnovationsStateSpace(
        torch.as_tensor(fixture.transition),
        torch.as_tensor(fixture.observation),
        converted.gain,
        converted.covariance,
    )
    ct_transfer = ct.innovations_transfer_function(innovations, frequencies_t)
    ct_transfer = ct_transfer.detach().cpu().numpy()
    cb_transfer = np.transpose(
        mvgc.ss2trfun(
            fixture.transition,
            fixture.observation,
            converted.gain.detach().cpu().numpy(),
            fres,
        ),
        (2, 0, 1),
    )
    cb_inverse = np.transpose(
        mvgc.ss2itrfun(
            fixture.transition,
            fixture.observation,
            converted.gain.detach().cpu().numpy(),
            fres,
        ),
        (2, 0, 1),
    )

    report.compare(
        "general_ssm_ct_transfer_vs_independent", ct_transfer, reference, atol=5e-10
    )
    report.compare(
        "general_ssm_complexbox_transfer_vs_independent",
        cb_transfer,
        reference,
        atol=5e-10,
    )
    report.compare(
        "general_ssm_ct_vs_complexbox_transfer",
        ct_transfer,
        cb_transfer,
        atol=5e-10,
    )
    identity_products = np.stack(
        [cb_inverse[index] @ cb_transfer[index] for index in range(fres + 1)]
    )
    identity = np.broadcast_to(
        np.eye(fixture.observation.shape[0]), identity_products.shape
    )
    report.compare(
        "general_ssm_transfer_inverse_identity",
        identity_products,
        identity,
        atol=6e-10,
    )

    covariance = converted.covariance.detach().cpu().numpy()
    reference_spectrum = spectrum_from_transfer(reference, covariance)
    ct_spectrum = spectrum_from_transfer(ct_transfer, covariance)
    cb_spectrum = np.transpose(
        mvgc.ss_to_cpsd(
            fixture.transition,
            fixture.observation,
            converted.gain.detach().cpu().numpy(),
            covariance,
            fres,
        ),
        (2, 0, 1),
    )
    report.compare(
        "general_ssm_ct_spectrum_vs_independent",
        ct_spectrum,
        reference_spectrum,
        atol=9e-10,
    )
    report.compare(
        "general_ssm_complexbox_spectrum_vs_independent",
        cb_spectrum,
        reference_spectrum,
        atol=9e-10,
    )
    _spectral_diagnostics(report, "general_ssm", reference_spectrum)
    gamma0 = general_ssm_autocovariances_reference(fixture, 0)[0]
    report.compare(
        "general_ssm_wiener_khinchin_gamma0",
        _integrated_gamma0(reference_spectrum, frequencies),
        gamma0,
        atol=3e-9,
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
