import math

import numpy as np
import torch
from scipy.linalg import solve_discrete_lyapunov
from scipy.optimize import brentq

from complextorch.measures.pid import gaussian_pid_from_var
from complextorch.numit import (
    _match_tmi_by_spectral_radius,
    _random_var_shapes,
    _wishart_identity,
    var_total_mutual_information,
)
from complextorch.representations import build_var_system


def _reference_companion(coefficients: np.ndarray) -> np.ndarray:
    order, n_variables, _ = coefficients.shape
    companion = np.zeros((order * n_variables, order * n_variables))
    companion[:n_variables] = coefficients.transpose(1, 0, 2).reshape(
        n_variables, order * n_variables
    )
    if order > 1:
        companion[n_variables:, :-n_variables] = np.eye(
            (order - 1) * n_variables
        )
    return companion


def _reference_scale_to_radius(
    coefficients: np.ndarray, radius: float
) -> np.ndarray:
    old_radius = np.abs(np.linalg.eigvals(_reference_companion(coefficients))).max()
    factor = radius / old_radius
    return np.stack(
        [coefficients[lag] * factor ** (lag + 1) for lag in range(coefficients.shape[0])]
    )


def _reference_state_covariance(
    coefficients: np.ndarray, innovation_covariance: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    companion = _reference_companion(coefficients)
    n_variables = innovation_covariance.shape[0]
    noise = np.zeros_like(companion)
    noise[:n_variables, :n_variables] = innovation_covariance
    return companion, solve_discrete_lyapunov(companion, noise)


def _reference_tmi(
    coefficients: np.ndarray, innovation_covariance: np.ndarray
) -> float:
    _, state_covariance = _reference_state_covariance(
        coefficients, innovation_covariance
    )
    n_variables = innovation_covariance.shape[0]
    present_covariance = state_covariance[:n_variables, :n_variables]
    return float(
        0.5
        * (
            np.linalg.slogdet(present_covariance)[1]
            - np.linalg.slogdet(innovation_covariance)[1]
        )
        / math.log(2.0)
    )


def _reference_gaussian_mi(covariance: np.ndarray, n_left: int) -> float:
    return float(
        0.5
        * (
            np.linalg.slogdet(covariance[:n_left, :n_left])[1]
            + np.linalg.slogdet(covariance[n_left:, n_left:])[1]
            - np.linalg.slogdet(covariance)[1]
        )
        / math.log(2.0)
    )


def _reference_mmi_pid(
    coefficients: np.ndarray, innovation_covariance: np.ndarray
) -> np.ndarray:
    companion, state_covariance = _reference_state_covariance(
        coefficients, innovation_covariance
    )
    order = coefficients.shape[0]
    n_variables = coefficients.shape[1]
    half = n_variables // 2
    source0 = [
        lag * n_variables + variable
        for lag in range(order)
        for variable in range(half)
    ]
    source1 = [
        lag * n_variables + variable
        for lag in range(order)
        for variable in range(half, n_variables)
    ]
    history = source0 + source1
    history_covariance = state_covariance[np.ix_(history, history)]
    projection = np.zeros((n_variables, order * n_variables))
    projection[:, :n_variables] = np.eye(n_variables)
    history_future = (
        state_covariance @ companion.T @ projection.T
    )[history]
    future_covariance = state_covariance[:n_variables, :n_variables]
    joint = np.block(
        [
            [history_covariance, history_future],
            [history_future.T, future_covariance],
        ]
    )
    n0 = len(source0)
    n1 = len(source1)
    target = list(range(n0 + n1, n0 + n1 + n_variables))

    def marginal_mi(source: list[int]) -> float:
        indices = source + target
        block = joint[np.ix_(indices, indices)]
        return _reference_gaussian_mi(block, len(source))

    i0 = marginal_mi(list(range(n0)))
    i1 = marginal_mi(list(range(n0, n0 + n1)))
    total = _reference_gaussian_mi(joint, n0 + n1)
    redundancy = min(i0, i1)
    unique0 = i0 - redundancy
    unique1 = i1 - redundancy
    synergy = total - redundancy - unique0 - unique1
    return np.array([redundancy, unique0, unique1, synergy, total])


def _reference_match(
    coefficients: np.ndarray,
    innovation_covariance: np.ndarray,
    target_tmi: float,
) -> tuple[float, np.ndarray]:
    def objective(radius: float) -> float:
        scaled = _reference_scale_to_radius(coefficients, radius)
        return _reference_tmi(scaled, innovation_covariance) - target_tmi

    low = 1e-10
    for gap in (
        1e-1,
        5e-2,
        1e-2,
        1e-3,
        1e-4,
        1e-5,
        1e-6,
        1e-7,
        1e-8,
        1e-9,
        1e-10,
    ):
        high = 1.0 - gap
        try:
            upper = objective(high)
        except Exception:
            continue
        if upper >= 0:
            radius = brentq(
                objective, low, high, xtol=1e-13, rtol=1e-13, maxiter=200
            )
            scaled = _reference_scale_to_radius(coefficients, radius)
            return radius, _reference_mmi_pid(scaled, innovation_covariance)
    raise RuntimeError("reference target TMI was not bracketed")


def test_large_null_ensemble_matches_high_tmi_without_unit_root_batch_failure():
    generator = torch.Generator().manual_seed(901)
    batch = 256
    coefficients = _random_var_shapes(
        batch,
        2,
        1,
        generator=generator,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    covariance = _wishart_identity(
        batch,
        2,
        generator=generator,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    target = torch.full((batch,), 15.0, dtype=torch.float64)
    model, radius = _match_tmi_by_spectral_radius(
        coefficients, covariance, target, base=2.0
    )
    achieved = var_total_mutual_information(model)
    torch.testing.assert_close(achieved, target, rtol=2e-6, atol=2e-5)
    assert bool((radius > 0.999999).any())
    assert bool((radius < 1.0).all())


def test_batched_var2_ensemble_matches_tmi_across_network_size():
    generator = torch.Generator().manual_seed(902)
    coefficients = _random_var_shapes(
        64,
        8,
        2,
        generator=generator,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    covariance = _wishart_identity(
        64,
        8,
        generator=generator,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    target = torch.full((64,), 8.0, dtype=torch.float64)
    model, _ = _match_tmi_by_spectral_radius(
        coefficients, covariance, target, base=2.0
    )
    torch.testing.assert_close(
        var_total_mutual_information(model), target, rtol=1e-9, atol=1e-9
    )


def test_numit_var_ensemble_matches_independent_scipy_reference():
    rng = np.random.default_rng(903)
    batch = 8
    n_variables = 4
    order = 2
    target = 2.0
    coefficients = rng.standard_normal((batch, order, n_variables, n_variables))
    normal = rng.standard_normal((batch, n_variables, n_variables + 1))
    covariance = normal @ np.swapaxes(normal, -1, -2)

    torch_model, torch_radius = _match_tmi_by_spectral_radius(
        torch.tensor(coefficients, dtype=torch.float64),
        torch.tensor(covariance, dtype=torch.float64),
        torch.full((batch,), target, dtype=torch.float64),
        base=2.0,
    )
    torch_pid = gaussian_pid_from_var(
        torch_model, (0, 1), (2, 3), redundancy="mmi"
    )
    torch_atoms = torch.stack(
        [
            torch_pid["redundant"],
            torch_pid["unique_source0"],
            torch_pid["unique_source1"],
            torch_pid["synergistic"],
            torch_pid["total"],
        ],
        dim=-1,
    ).numpy()

    reference_radius = []
    reference_atoms = []
    for index in range(batch):
        radius, atoms = _reference_match(
            coefficients[index], covariance[index], target
        )
        reference_radius.append(radius)
        reference_atoms.append(atoms)

    np.testing.assert_allclose(
        torch_radius.numpy(), np.asarray(reference_radius), rtol=1e-10, atol=1e-11
    )
    np.testing.assert_allclose(
        torch_atoms, np.asarray(reference_atoms), rtol=1e-9, atol=1e-10
    )
