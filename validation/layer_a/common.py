"""Shared fixtures, analytical references, and reporting for Layer A parity."""
from __future__ import annotations

import csv
import json
import math
import os
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov

COMPLEXBOX_COMMIT = "87b5e2cd9bba22ddd978bade6f614da7d6190db2"
DEFAULT_ATOL = 1e-9
STRICT_ATOL = 1e-11

PASS_EXACT = "PASS_EXACT"
PASS_NUMERICAL = "PASS_NUMERICAL"
PASS_EQUIVALENT = "PASS_EQUIVALENT_REPRESENTATION"
EXPECTED_DIFFERENCE = "EXPECTED_DIFFERENCE_CONVENTION"
FAIL = "FAIL"


@dataclass(frozen=True)
class CheckResult:
    suite: str
    check: str
    status: str
    metric: str
    observed: float | str | None = None
    reference: float | str | None = None
    absolute_error: float | None = None
    relative_error: float | None = None
    tolerance: float | None = None
    notes: str = ""

    @property
    def failed(self) -> bool:
        return self.status == FAIL


class Report:
    """Collect machine-readable validation results without hiding discrepancies."""

    def __init__(self, suite: str) -> None:
        self.suite = suite
        self.results: list[CheckResult] = []

    def add(self, result: CheckResult) -> CheckResult:
        self.results.append(result)
        print(
            f"[{result.status}] {result.suite} :: {result.check} "
            f"({result.metric}; abs={result.absolute_error}, tol={result.tolerance})"
        )
        if result.notes:
            print(f"  {result.notes}")
        return result

    def compare(
        self,
        check: str,
        observed: Any,
        reference: Any,
        *,
        atol: float = DEFAULT_ATOL,
        metric: str = "max_abs_error",
        pass_status: str = PASS_NUMERICAL,
        notes: str = "",
    ) -> CheckResult:
        obs = np.asarray(observed)
        ref = np.asarray(reference)
        if obs.shape != ref.shape:
            return self.add(
                CheckResult(
                    self.suite,
                    check,
                    FAIL,
                    "shape",
                    str(obs.shape),
                    str(ref.shape),
                    tolerance=0.0,
                    notes=notes,
                )
            )
        if not np.all(np.isfinite(obs)) or not np.all(np.isfinite(ref)):
            return self.add(
                CheckResult(
                    self.suite,
                    check,
                    FAIL,
                    "finite_values",
                    str(bool(np.all(np.isfinite(obs)))),
                    str(bool(np.all(np.isfinite(ref)))),
                    notes=notes,
                )
            )
        absolute_error = float(np.max(np.abs(obs - ref))) if obs.size else 0.0
        scale = float(np.max(np.abs(ref))) if ref.size else 0.0
        relative_error = absolute_error / max(scale, np.finfo(float).tiny)
        return self.add(
            CheckResult(
                self.suite,
                check,
                pass_status if absolute_error <= atol else FAIL,
                metric,
                observed=float(np.max(np.abs(obs))) if obs.size else 0.0,
                reference=float(np.max(np.abs(ref))) if ref.size else 0.0,
                absolute_error=absolute_error,
                relative_error=relative_error,
                tolerance=atol,
                notes=notes,
            )
        )

    def scalar_bound(
        self,
        check: str,
        value: float,
        *,
        upper: float,
        metric: str,
        notes: str = "",
        pass_status: str = PASS_NUMERICAL,
    ) -> CheckResult:
        value = float(value)
        return self.add(
            CheckResult(
                self.suite,
                check,
                pass_status if math.isfinite(value) and value <= upper else FAIL,
                metric,
                observed=value,
                reference=0.0,
                absolute_error=abs(value),
                relative_error=None,
                tolerance=upper,
                notes=notes,
            )
        )

    def expected_difference(
        self,
        check: str,
        observed: Any,
        reference: Any,
        *,
        metric: str,
        notes: str,
    ) -> CheckResult:
        obs = np.asarray(observed)
        ref = np.asarray(reference)
        error = float(np.max(np.abs(obs - ref))) if obs.shape == ref.shape and obs.size else None
        return self.add(
            CheckResult(
                self.suite,
                check,
                EXPECTED_DIFFERENCE,
                metric,
                absolute_error=error,
                notes=notes,
            )
        )

    def write(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = [asdict(item) for item in self.results]
        (output_dir / f"{self.suite}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        fieldnames = list(asdict(self.results[0]).keys()) if self.results else list(
            CheckResult.__dataclass_fields__.keys()
        )
        with (output_dir / f"{self.suite}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(payload)

    def raise_for_failures(self) -> None:
        failures = [item for item in self.results if item.failed]
        if failures:
            names = ", ".join(item.check for item in failures)
            raise AssertionError(f"{self.suite}: {len(failures)} failed check(s): {names}")


@dataclass(frozen=True)
class VARFixture:
    name: str
    coefficients: np.ndarray
    innovation_covariance: np.ndarray

    @property
    def complexbox_coefficients(self) -> np.ndarray:
        return np.transpose(self.coefficients, (1, 2, 0)).copy()


@dataclass(frozen=True)
class GeneralSSMFixture:
    name: str
    transition: np.ndarray
    observation: np.ndarray
    process_covariance: np.ndarray
    observation_covariance: np.ndarray


@dataclass(frozen=True)
class InnovationsFixture:
    name: str
    transition: np.ndarray
    observation: np.ndarray
    gain: np.ndarray
    innovation_covariance: np.ndarray


def var_fixtures() -> tuple[VARFixture, ...]:
    triangular = VARFixture(
        "var1_triangular",
        np.array(
            [[[0.55, 0.00, 0.00], [0.35, 0.45, 0.00], [0.00, 0.25, 0.35]]],
            dtype=np.float64,
        ),
        np.diag([1.0, 0.8, 0.6]).astype(np.float64),
    )
    higher_order = VARFixture(
        "var3_correlated_noise",
        np.array(
            [
                [[0.40, 0.15, 0.00], [0.00, 0.35, 0.10], [0.00, 0.00, 0.30]],
                [[-0.12, 0.00, 0.00], [0.08, -0.10, 0.00], [0.00, 0.05, -0.08]],
                [[0.05, 0.00, 0.00], [0.00, 0.04, 0.00], [0.03, 0.00, 0.02]],
            ],
            dtype=np.float64,
        ),
        np.array(
            [[1.00, 0.20, 0.10], [0.20, 0.80, 0.15], [0.10, 0.15, 0.60]],
            dtype=np.float64,
        ),
    )
    return triangular, higher_order


def general_ssm_fixture() -> GeneralSSMFixture:
    return GeneralSSMFixture(
        "general_ssm_observable",
        np.array(
            [[0.72, -0.18, 0.00], [0.18, 0.68, 0.10], [0.00, -0.08, 0.52]],
            dtype=np.float64,
        ),
        np.array([[1.00, 0.20, 0.00], [0.00, 0.35, 1.00]], dtype=np.float64),
        np.array(
            [[0.20, 0.03, 0.01], [0.03, 0.15, 0.02], [0.01, 0.02, 0.10]],
            dtype=np.float64,
        ),
        np.array([[0.12, 0.025], [0.025, 0.09]], dtype=np.float64),
    )


def companion_matrix_np(coefficients: np.ndarray) -> np.ndarray:
    coefficients = np.asarray(coefficients, dtype=np.float64)
    order, n_variables, n_sources = coefficients.shape
    if n_variables != n_sources:
        raise ValueError("VAR coefficient blocks must be square")
    companion = np.zeros((order * n_variables, order * n_variables), dtype=np.float64)
    companion[:n_variables, :] = np.transpose(coefficients, (1, 0, 2)).reshape(
        n_variables, order * n_variables
    )
    if order > 1:
        companion[n_variables:, : (order - 1) * n_variables] = np.eye(
            (order - 1) * n_variables
        )
    return companion


def projection_np(n_variables: int, order: int) -> np.ndarray:
    projection = np.zeros((n_variables, n_variables * order), dtype=np.float64)
    projection[:, :n_variables] = np.eye(n_variables)
    return projection


def spectral_radius_np(matrix: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(np.asarray(matrix)))))


def var_stationary_state_covariance(fixture: VARFixture) -> np.ndarray:
    companion = companion_matrix_np(fixture.coefficients)
    n_variables = fixture.coefficients.shape[1]
    noise = np.zeros_like(companion)
    noise[:n_variables, :n_variables] = fixture.innovation_covariance
    return solve_discrete_lyapunov(companion, noise)


def general_ssm_state_covariance(fixture: GeneralSSMFixture) -> np.ndarray:
    return solve_discrete_lyapunov(fixture.transition, fixture.process_covariance)


def dare_reference(fixture: GeneralSSMFixture) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prediction_covariance = solve_discrete_are(
        fixture.transition.T,
        fixture.observation.T,
        fixture.process_covariance,
        fixture.observation_covariance,
    )
    prediction_covariance = 0.5 * (prediction_covariance + prediction_covariance.T)
    innovation_covariance = (
        fixture.observation @ prediction_covariance @ fixture.observation.T
        + fixture.observation_covariance
    )
    gain = (
        fixture.transition
        @ prediction_covariance
        @ fixture.observation.T
        @ np.linalg.solve(innovation_covariance, np.eye(innovation_covariance.shape[0]))
    )
    return prediction_covariance, gain, innovation_covariance


def innovations_from_general_ssm(fixture: GeneralSSMFixture) -> InnovationsFixture:
    _, gain, innovation_covariance = dare_reference(fixture)
    return InnovationsFixture(
        f"{fixture.name}_innovations",
        fixture.transition,
        fixture.observation,
        gain,
        innovation_covariance,
    )


def innovations_from_var(fixture: VARFixture) -> InnovationsFixture:
    coefficients = fixture.coefficients
    order, n_variables, _ = coefficients.shape
    transition = companion_matrix_np(coefficients)
    observation = np.transpose(coefficients, (1, 0, 2)).reshape(
        n_variables, order * n_variables
    )
    gain = np.zeros((order * n_variables, n_variables), dtype=np.float64)
    gain[:n_variables, :] = np.eye(n_variables)
    return InnovationsFixture(
        f"{fixture.name}_innovations",
        transition,
        observation,
        gain,
        fixture.innovation_covariance,
    )


def var_autocovariances_reference(fixture: VARFixture, max_lag: int) -> np.ndarray:
    companion = companion_matrix_np(fixture.coefficients)
    covariance = var_stationary_state_covariance(fixture)
    projection = projection_np(fixture.coefficients.shape[1], fixture.coefficients.shape[0])
    result = []
    power = np.eye(companion.shape[0])
    for _ in range(max_lag + 1):
        result.append(projection @ power @ covariance @ projection.T)
        power = companion @ power
    return np.stack(result, axis=0)


def general_ssm_autocovariances_reference(
    fixture: GeneralSSMFixture, max_lag: int
) -> np.ndarray:
    covariance = general_ssm_state_covariance(fixture)
    result = [
        fixture.observation @ covariance @ fixture.observation.T
        + fixture.observation_covariance
    ]
    power = fixture.transition.copy()
    for _ in range(1, max_lag + 1):
        result.append(fixture.observation @ power @ covariance @ fixture.observation.T)
        power = fixture.transition @ power
    return np.stack(result, axis=0)


def innovations_autocovariances_reference(
    fixture: InnovationsFixture, max_lag: int
) -> np.ndarray:
    omega = solve_discrete_lyapunov(
        fixture.transition,
        fixture.gain @ fixture.innovation_covariance @ fixture.gain.T,
    )
    gamma0 = (
        fixture.observation @ omega @ fixture.observation.T
        + fixture.innovation_covariance
    )
    result = [gamma0]
    if max_lag == 0:
        return np.stack(result, axis=0)
    lam = (
        fixture.transition @ omega @ fixture.observation.T
        + fixture.gain @ fixture.innovation_covariance
    )
    result.append(fixture.observation @ lam)
    for _ in range(2, max_lag + 1):
        lam = fixture.transition @ lam
        result.append(fixture.observation @ lam)
    return np.stack(result, axis=0)


def var_transfer_reference(fixture: VARFixture, frequencies: np.ndarray) -> np.ndarray:
    n_variables = fixture.coefficients.shape[1]
    identity = np.eye(n_variables, dtype=np.complex128)
    output = []
    for frequency in np.asarray(frequencies, dtype=np.float64):
        omega = 2.0 * np.pi * frequency
        polynomial = identity.copy()
        for lag, coefficient in enumerate(fixture.coefficients, start=1):
            polynomial -= coefficient * np.exp(-1j * omega * lag)
        output.append(np.linalg.solve(polynomial, identity))
    return np.stack(output, axis=0)


def innovations_transfer_reference(
    fixture: InnovationsFixture, frequencies: np.ndarray
) -> np.ndarray:
    state_identity = np.eye(fixture.transition.shape[0], dtype=np.complex128)
    observation_identity = np.eye(fixture.observation.shape[0], dtype=np.complex128)
    output = []
    for frequency in np.asarray(frequencies, dtype=np.float64):
        z = np.exp(2j * np.pi * frequency)
        output.append(
            observation_identity
            + fixture.observation
            @ np.linalg.solve(z * state_identity - fixture.transition, fixture.gain)
        )
    return np.stack(output, axis=0)


def spectrum_from_transfer(transfer: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    return np.stack(
        [matrix @ covariance @ matrix.conj().T for matrix in np.asarray(transfer)],
        axis=0,
    )


def covariance_to_block_toeplitz(autocovariances: np.ndarray) -> np.ndarray:
    autocovariances = np.asarray(autocovariances)
    count, n_variables, _ = autocovariances.shape
    output = np.empty((count * n_variables, count * n_variables), dtype=autocovariances.dtype)
    for row in range(count):
        for column in range(count):
            lag = row - column
            block = autocovariances[lag] if lag >= 0 else autocovariances[-lag].T
            output[
                row * n_variables : (row + 1) * n_variables,
                column * n_variables : (column + 1) * n_variables,
            ] = block
    return output


def environment_payload() -> dict[str, Any]:
    import scipy

    payload: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "complexbox_pinned_commit": COMPLEXBOX_COMMIT,
        "complextorch_git_sha": os.environ.get("GITHUB_SHA", "unknown"),
    }
    try:
        import torch

        payload["torch"] = torch.__version__
        payload["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:
        payload["torch_error"] = repr(exc)
    for package_name in ("complextorch", "complexbox"):
        try:
            module = __import__(package_name)
            payload[package_name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            payload[f"{package_name}_error"] = repr(exc)
    return payload


def aggregate_results(paths: Iterable[Path], output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "layer_a_summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
    )
    if rows:
        with (output_dir / "layer_a_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return rows
