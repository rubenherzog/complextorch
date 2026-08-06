#!/usr/bin/env python3
"""Strict Layer A parity using only installed repository packages.

This script must run after independently checking out and installing the exact
ComplexTorch and ComplexBox repositories. Each toolbox computes its own outputs.
No output from one toolbox is used as an input to the other. Every quantity is
compared in the triangle ComplexTorch–ComplexBox–independent reference.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov

import complextorch as ct
from complextorch.linalg import solve_discrete_lyapunov as ct_dlyap
from complexbox import mvgc

ATOL_EXACT = 0.0
ATOL_TIGHT = 3e-10
ATOL_ITERATIVE = 5e-9


def _np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _maxerr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return math.inf
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def _companion(coefficients):
    p, n, _ = coefficients.shape
    out = np.zeros((p * n, p * n), dtype=float)
    out[:n, :] = np.transpose(coefficients, (1, 0, 2)).reshape(n, p * n)
    if p > 1:
        out[n:, : (p - 1) * n] = np.eye((p - 1) * n)
    return out


def _projection(n, p):
    c = np.zeros((n, n * p), dtype=float)
    c[:, :n] = np.eye(n)
    return c


def _var_autocov_reference(coefficients, sigma, max_lag):
    a = _companion(coefficients)
    n = coefficients.shape[1]
    q = np.zeros_like(a)
    q[:n, :n] = sigma
    p = solve_discrete_lyapunov(a, q)
    c = _projection(n, coefficients.shape[0])
    values = []
    power = np.eye(a.shape[0])
    for _ in range(max_lag + 1):
        values.append(c @ power @ p @ c.T)
        power = a @ power
    return np.stack(values)


def _var_transfer_reference(coefficients, frequencies):
    n = coefficients.shape[1]
    eye = np.eye(n, dtype=complex)
    values = []
    for f in frequencies:
        polynomial = eye.copy()
        for lag, block in enumerate(coefficients, start=1):
            polynomial -= block * np.exp(-2j * np.pi * f * lag)
        values.append(np.linalg.solve(polynomial, eye))
    return np.stack(values)


def _iss_transfer_reference(a, c, k, frequencies):
    eye_x = np.eye(a.shape[0], dtype=complex)
    eye_y = np.eye(c.shape[0], dtype=complex)
    values = []
    for f in frequencies:
        z = np.exp(2j * np.pi * f)
        values.append(eye_y + c @ np.linalg.solve(z * eye_x - a, k))
    return np.stack(values)


def _general_ssm_autocov_reference(a, c, q, r, max_lag):
    p = solve_discrete_lyapunov(a, q)
    values = [c @ p @ c.T + r]
    power = a.copy()
    for _ in range(1, max_lag + 1):
        values.append(c @ power @ p @ c.T)
        power = a @ power
    return np.stack(values)


def _riccati_residual(a, c, q, r, p, s=None):
    s = np.zeros((a.shape[0], c.shape[0])) if s is None else s
    v = c @ p @ c.T + r
    u = a @ p @ c.T + s
    return p - a @ p @ a.T - q + u @ np.linalg.solve(v, u.T)


class Audit:
    def __init__(self):
        self.rows = []

    def compare(self, quantity, left_name, left, right_name, right, atol):
        error = _maxerr(left, right)
        passed = bool(np.isfinite(error) and error <= atol)
        self.rows.append({
            "quantity": quantity,
            "comparison": f"{left_name}_vs_{right_name}",
            "max_abs_error": error,
            "tolerance": atol,
            "passed": passed,
            "left_shape": list(np.asarray(left).shape),
            "right_shape": list(np.asarray(right).shape),
        })
        print(f"{'PASS' if passed else 'FAIL'} {quantity} {left_name} vs {right_name}: {error:.3e} <= {atol:.3e}")

    def triplet(self, quantity, ct_value, cb_value, reference, atol):
        self.compare(quantity, "complextorch", ct_value, "complexbox", cb_value, atol)
        self.compare(quantity, "complextorch", ct_value, "reference", reference, atol)
        self.compare(quantity, "complexbox", cb_value, "reference", reference, atol)

    def bound(self, quantity, value, upper):
        value = float(value)
        passed = bool(np.isfinite(value) and value <= upper)
        self.rows.append({
            "quantity": quantity,
            "comparison": "residual_bound",
            "max_abs_error": value,
            "tolerance": upper,
            "passed": passed,
            "left_shape": [],
            "right_shape": [],
        })
        print(f"{'PASS' if passed else 'FAIL'} {quantity}: {value:.3e} <= {upper:.3e}")

    def finish(self, output):
        output.mkdir(parents=True, exist_ok=True)
        metadata = {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "complextorch_file": ct.__file__,
            "complexbox_file": __import__("complexbox").__file__,
            "github_sha": os.environ.get("GITHUB_SHA"),
            "complexbox_expected_commit": "87b5e2cd9bba22ddd978bade6f614da7d6190db2",
            "rows": self.rows,
        }
        (output / "installed_repo_triplets.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        failures = [row for row in self.rows if not row["passed"]]
        if failures:
            raise AssertionError(f"{len(failures)} strict triplet parity checks failed")


def main():
    output = Path(sys.argv[1])
    audit = Audit()

    var_cases = [
        (
            "var1",
            np.array([[[0.55, 0.00, 0.00], [0.35, 0.45, 0.00], [0.00, 0.25, 0.35]]]),
            np.diag([1.0, 0.8, 0.6]),
        ),
        (
            "var3",
            np.array([
                [[0.40, 0.15, 0.00], [0.00, 0.35, 0.10], [0.00, 0.00, 0.30]],
                [[-0.12, 0.00, 0.00], [0.08, -0.10, 0.00], [0.00, 0.05, -0.08]],
                [[0.05, 0.00, 0.00], [0.00, 0.04, 0.00], [0.03, 0.00, 0.02]],
            ]),
            np.array([[1.0, 0.2, 0.1], [0.2, 0.8, 0.15], [0.1, 0.15, 0.6]]),
        ),
    ]

    frequencies = np.linspace(0.0, 0.5, 513)
    for name, coefficients, sigma in var_cases:
        cb_coefficients = np.transpose(coefficients, (1, 2, 0))
        companion = _companion(coefficients)
        n = coefficients.shape[1]
        companion_q = np.zeros_like(companion)
        companion_q[:n, :n] = sigma

        ct_system = ct.build_var_system(torch.tensor(coefficients), torch.tensor(sigma))
        ct_iss = ct.var_to_innovations_state_space(ct_system)
        cb_a, cb_c, cb_k, cb_v = mvgc.var_to_ss(cb_coefficients, sigma)
        ref_c = np.transpose(coefficients, (1, 0, 2)).reshape(n, -1)
        ref_k = np.zeros((companion.shape[0], n)); ref_k[:n] = np.eye(n)

        audit.triplet(f"{name}.var_to_ss.A", _np(ct_iss.transition[0]), cb_a, companion, ATOL_EXACT)
        audit.triplet(f"{name}.var_to_ss.C", _np(ct_iss.observation[0]), cb_c, ref_c, ATOL_EXACT)
        audit.triplet(f"{name}.var_to_ss.K", _np(ct_iss.gain[0]), cb_k, ref_k, ATOL_EXACT)
        audit.triplet(f"{name}.var_to_ss.V", _np(ct_iss.innovation_covariance[0]), cb_v, sigma, ATOL_EXACT)

        ref_p = solve_discrete_lyapunov(companion, companion_q)
        ct_p_direct, _ = ct_dlyap(torch.tensor(companion), torch.tensor(companion_q), method="direct")
        ct_p_doubling, _ = ct_dlyap(torch.tensor(companion), torch.tensor(companion_q), method="doubling")
        cb_p = mvgc.dlyap(companion, companion_q)
        audit.triplet(f"{name}.lyapunov.direct", _np(ct_p_direct), cb_p, ref_p, ATOL_TIGHT)
        audit.triplet(f"{name}.lyapunov.doubling", _np(ct_p_doubling), cb_p, ref_p, ATOL_TIGHT)

        max_lag = 8
        ref_gamma = _var_autocov_reference(coefficients, sigma, max_lag)
        ct_gamma = _np(ct.model_autocovariances(ct_system, max_lag)[0])
        cb_gamma, _ = mvgc.var_to_autocov(cb_coefficients, sigma, qmax=-max_lag)
        cb_gamma = np.transpose(cb_gamma, (2, 0, 1))
        audit.triplet(f"{name}.autocov", ct_gamma, cb_gamma, ref_gamma, ATOL_TIGHT)

        ref_h = _var_transfer_reference(coefficients, frequencies)
        ct_h = _np(ct.innovations_transfer_function(ct_iss, torch.tensor(frequencies))[0])
        cb_h = np.transpose(mvgc.var2trfun(cb_coefficients, 512), (2, 0, 1))
        audit.triplet(f"{name}.transfer", ct_h, cb_h, ref_h, 5e-10)

        ref_s = np.stack([h @ sigma @ h.conj().T for h in ref_h])
        ct_s = np.stack([h @ sigma @ h.conj().T for h in ct_h])
        cb_s = np.transpose(mvgc.var_to_cpsd(cb_coefficients, sigma, 512), (2, 0, 1))
        audit.triplet(f"{name}.spectrum", ct_s, cb_s, ref_s, 1e-9)

    a = np.array([[0.72, -0.18, 0.00], [0.18, 0.68, 0.10], [0.00, -0.08, 0.52]])
    c = np.array([[1.00, 0.20, 0.00], [0.00, 0.35, 1.00]])
    q = np.array([[0.20, 0.03, 0.01], [0.03, 0.15, 0.02], [0.01, 0.02, 0.10]])
    r = np.array([[0.12, 0.025], [0.025, 0.09]])
    stationary_p = solve_discrete_lyapunov(a, q)
    model = ct.StateSpaceModel(torch.tensor(a), torch.tensor(c), torch.tensor(q), torch.tensor(r), torch.tensor(stationary_p))

    ref_p = solve_discrete_are(a.T, c.T, q, r)
    ref_p = 0.5 * (ref_p + ref_p.T)
    ref_v = c @ ref_p @ c.T + r
    ref_k = a @ ref_p @ c.T @ np.linalg.solve(ref_v, np.eye(ref_v.shape[0]))

    ct_conv = ct.innovations_form(model)
    cb_k, cb_v, cb_rep, cb_p = mvgc.mdare(a, c, q, r)
    audit.triplet("ssm.dare.P", _np(ct_conv.prediction_covariance), cb_p, ref_p, ATOL_TIGHT)
    audit.triplet("ssm.dare.K", _np(ct_conv.gain), cb_k, ref_k, ATOL_TIGHT)
    audit.triplet("ssm.dare.V", _np(ct_conv.covariance), cb_v, ref_v, ATOL_TIGHT)
    audit.bound("ssm.dare.ct_residual", np.max(np.abs(_riccati_residual(a, c, q, r, _np(ct_conv.prediction_covariance)))), ATOL_TIGHT)
    audit.bound("ssm.dare.cb_residual", np.max(np.abs(_riccati_residual(a, c, q, r, cb_p))), ATOL_TIGHT)
    audit.bound("ssm.dare.cb_reported_residual", cb_rep, ATOL_TIGHT)

    s = np.array([[0.015, 0.000], [0.005, 0.008], [0.000, 0.010]])
    ref_pg = solve_discrete_are(a.T, c.T, q, r, s=s)
    ref_pg = 0.5 * (ref_pg + ref_pg.T)
    ct_pg = _np(ct.solve_generalized_dare(torch.tensor(a), torch.tensor(c), torch.tensor(q), torch.tensor(r), torch.tensor(s)))
    _, _, cb_rep_g, cb_pg = mvgc.mdare(a, c, q, r, s)
    audit.triplet("ssm.generalized_dare.P", ct_pg, cb_pg, ref_pg, ATOL_ITERATIVE)
    audit.bound("ssm.generalized_dare.ct_residual", np.max(np.abs(_riccati_residual(a, c, q, r, ct_pg, s))), ATOL_ITERATIVE)
    audit.bound("ssm.generalized_dare.cb_residual", np.max(np.abs(_riccati_residual(a, c, q, r, cb_pg, s))), ATOL_TIGHT)
    audit.bound("ssm.generalized_dare.cb_reported_residual", cb_rep_g, ATOL_TIGHT)

    max_lag = 8
    ref_gamma = _general_ssm_autocov_reference(a, c, q, r, max_lag)
    ct_gamma = _np(ct.model_autocovariances(model, max_lag))
    cb_gamma, _ = mvgc.ss_to_autocov(a, c, cb_k, cb_v, qmax=-max_lag)
    cb_gamma = np.transpose(cb_gamma, (2, 0, 1))
    audit.triplet("ssm.autocov", ct_gamma, cb_gamma, ref_gamma, 5e-10)

    ct_iss = ct.InnovationsStateSpace(torch.tensor(a), torch.tensor(c), ct_conv.gain, ct_conv.covariance)
    ct_h = _np(ct.innovations_transfer_function(ct_iss, torch.tensor(frequencies)))
    cb_h = np.transpose(mvgc.ss2trfun(a, c, cb_k, 512), (2, 0, 1))
    ref_h = _iss_transfer_reference(a, c, ref_k, frequencies)
    audit.triplet("ssm.transfer", ct_h, cb_h, ref_h, 7e-10)

    ct_spectrum = np.stack([h @ _np(ct_conv.covariance) @ h.conj().T for h in ct_h])
    cb_spectrum = np.transpose(mvgc.ss_to_cpsd(a, c, cb_k, cb_v, 512), (2, 0, 1))
    ref_spectrum = np.stack([h @ ref_v @ h.conj().T for h in ref_h])
    audit.triplet("ssm.spectrum", ct_spectrum, cb_spectrum, ref_spectrum, 1.5e-9)

    audit.finish(output)


if __name__ == "__main__":
    main()
