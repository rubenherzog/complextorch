"""Parity tests for Rosas--Mediano emergence and the full-past extension."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from complextorch.control import var_to_innovations_state_space
from complextorch.dd import dynamical_dependence
from complextorch.linalg import spd_logdet, symmetrise
from complextorch.measures import (
    ModelMeasureConfig,
    compute_all_model_measures,
    emergence_from_model,
    model_autocovariances,
)
from complextorch.representations import build_var_system


REFERENCE_COMMIT = "ecf591aacb6d58996c903b51a2f945cd7f713a32"


def _numpy_gaussian_mi(a: np.ndarray, b: np.ndarray, cross: np.ndarray) -> float:
    joint = np.block([[a, cross], [cross.T, b]])
    signs = [np.linalg.slogdet(matrix)[0] for matrix in (a, b, joint)]
    assert all(sign > 0 for sign in signs)
    return 0.5 * (
        np.linalg.slogdet(a)[1]
        + np.linalg.slogdet(b)[1]
        - np.linalg.slogdet(joint)[1]
    ) / np.log(2.0)


def _matlab_equation_reference(
    gamma0: np.ndarray, gamma_tau: np.ndarray, projection: np.ndarray
) -> dict[str, float]:
    """Eq. (10) / MATLAB reference, written independently with NumPy."""
    n = gamma0.shape[0]
    macro_cov = projection @ gamma0 @ projection.T
    macro_cross = projection @ gamma_tau @ projection.T
    macro_mi = _numpy_gaussian_mi(macro_cov, macro_cov, macro_cross.T)

    micro_to_macro = []
    for i in range(n):
        micro_to_macro.append(
            _numpy_gaussian_mi(
                gamma0[i : i + 1, i : i + 1],
                macro_cov,
                gamma_tau[:, i : i + 1].T @ projection.T,
            )
        )

    macro_to_micro = []
    micro_pairwise = np.empty((n, n))
    for j in range(n):
        macro_to_micro.append(
            _numpy_gaussian_mi(
                macro_cov,
                gamma0[j : j + 1, j : j + 1],
                projection @ gamma_tau[j : j + 1].T,
            )
        )
        for i in range(n):
            micro_pairwise[j, i] = _numpy_gaussian_mi(
                gamma0[i : i + 1, i : i + 1],
                gamma0[j : j + 1, j : j + 1],
                gamma_tau[j : j + 1, i : i + 1].T,
            )

    macro_to_micro = np.asarray(macro_to_micro)
    return {
        "psi": macro_mi - np.sum(micro_to_macro),
        "delta": np.max(macro_to_micro - micro_pairwise.sum(axis=1)),
        "gamma": np.max(macro_to_micro),
    }


def _system(dtype: torch.dtype = torch.float64):
    coefficients = torch.tensor(
        [
            [
                [0.45, 0.18, 0.00],
                [0.00, 0.35, 0.16],
                [0.08, 0.00, 0.28],
            ],
            [
                [-0.10, 0.00, 0.00],
                [0.04, -0.08, 0.00],
                [0.00, 0.03, -0.07],
            ],
        ],
        dtype=dtype,
    )
    covariance = torch.tensor(
        [[1.0, 0.12, 0.04], [0.12, 0.9, 0.08], [0.04, 0.08, 1.1]],
        dtype=dtype,
    )
    return build_var_system(coefficients, covariance)


def _projection(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.tensor(
        [[1.0, 1.0, 0.0], [0.0, 0.25, 1.0]], dtype=dtype
    ) / torch.tensor([[math.sqrt(2.0)], [math.sqrt(1.0625)]], dtype=dtype)


def _finite_history_mi(
    gamma: torch.Tensor,
    history_projection: torch.Tensor,
    target_projection: torch.Tensor,
    history_length: int,
) -> torch.Tensor:
    """I(projected finite past; current target) from block autocovariances."""
    rows = []
    for left in range(history_length):
        blocks = []
        for right in range(history_length):
            delta = right - left
            covariance = gamma[delta] if delta >= 0 else gamma[-delta].T
            blocks.append(history_projection @ covariance @ history_projection.T)
        rows.append(torch.cat(blocks, dim=-1))
    history_covariance = torch.cat(rows, dim=-2)
    cross = torch.cat(
        [
            target_projection @ gamma[lag] @ history_projection.T
            for lag in range(1, history_length + 1)
        ],
        dim=-1,
    )
    target_covariance = target_projection @ gamma[0] @ target_projection.T
    conditional = symmetrise(
        target_covariance
        - cross @ torch.linalg.solve(history_covariance, cross.T)
    )
    return 0.5 * (
        spd_logdet(target_covariance) - spd_logdet(conditional)
    ) / math.log(2.0)


def _finite_history_emergence_reference(
    system, projection: torch.Tensor, history_length: int = 10
) -> dict[str, torch.Tensor]:
    """Independent finite-history approximation to the full-past extension."""
    gamma = model_autocovariances(system, history_length)[0]
    n = gamma.shape[-1]
    eye = torch.eye(n, dtype=gamma.dtype, device=gamma.device)
    macro_mi = _finite_history_mi(gamma, projection, projection, history_length)
    micro_to_macro = torch.stack(
        [
            _finite_history_mi(
                gamma, eye[index : index + 1], projection, history_length
            )
            for index in range(n)
        ]
    )
    macro_to_micro = torch.stack(
        [
            _finite_history_mi(
                gamma, projection, eye[index : index + 1], history_length
            )
            for index in range(n)
        ]
    )
    pairwise = torch.stack(
        [
            torch.stack(
                [
                    _finite_history_mi(
                        gamma,
                        eye[source : source + 1],
                        eye[target : target + 1],
                        history_length,
                    )
                    for source in range(n)
                ]
            )
            for target in range(n)
        ]
    )
    return {
        "psi": macro_mi - micro_to_macro.sum(),
        "delta": (macro_to_micro - pairwise.sum(-1)).max(),
        "gamma": macro_to_micro.max(),
    }


@pytest.mark.parametrize("lag", [1, 2, 3])
def test_reconciling_emergences_matches_matlab_equations(lag: int) -> None:
    system = _system()
    projection = _projection()
    gamma = model_autocovariances(system, lag)[0]
    expected = _matlab_equation_reference(
        gamma[0].numpy(), gamma[lag].numpy(), projection.numpy()
    )
    actual = emergence_from_model(system, projection, lag=lag)
    for name in ("psi", "delta", "gamma"):
        np.testing.assert_allclose(
            actual[name].detach().cpu().numpy(), expected[name], rtol=1e-10, atol=1e-11
        )


def test_reconciling_emergences_batch_matches_individual_models() -> None:
    first = _system()
    second_coefficients = first.coefficients * 0.78
    coefficients = torch.cat([first.coefficients, second_coefficients], dim=0)
    covariance = first.innovation_covariance.expand(2, -1, -1).clone()
    batched_system = build_var_system(coefficients, covariance)
    projections = torch.stack([_projection(), _projection() * 1.3])

    batched = emergence_from_model(batched_system, projections, lag=2)
    for index in range(2):
        single_system = build_var_system(coefficients[index], covariance[index])
        single = emergence_from_model(single_system, projections[index], lag=2)
        for name in ("psi", "delta", "gamma"):
            torch.testing.assert_close(batched[name][index], single[name][0])


def test_reconciling_emergences_preserves_float32_and_agrees_with_float64() -> None:
    value32 = emergence_from_model(_system(torch.float32), _projection(torch.float32))
    value64 = emergence_from_model(_system(torch.float64), _projection(torch.float64))
    for name in ("psi", "delta", "gamma"):
        assert value32[name].dtype == torch.float32
        torch.testing.assert_close(
            value32[name].double(), value64[name], rtol=2e-5, atol=2e-6
        )


def test_var_and_equivalent_iss_have_identical_emergence() -> None:
    system = _system()
    innovations = var_to_innovations_state_space(system)
    projection = _projection()
    for history in ("lagged", "full"):
        var_value = emergence_from_model(system, projection, lag=3, history=history)
        iss_value = emergence_from_model(innovations, projection, lag=3, history=history)
        for name in ("psi", "delta", "gamma"):
            torch.testing.assert_close(
                iss_value[name], var_value[name], rtol=1e-10, atol=1e-11
            )


def test_independent_process_and_macro_equal_micro_have_known_behavior() -> None:
    coefficients = torch.diag(
        torch.tensor([0.55, 0.35, 0.20], dtype=torch.float64)
    )[None]
    covariance = torch.diag(torch.tensor([1.0, 0.8, 1.2], dtype=torch.float64))
    system = build_var_system(coefficients, covariance)
    projection = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    for history in ("lagged", "full"):
        value = emergence_from_model(system, projection, history=history)
        torch.testing.assert_close(
            value["psi"], torch.zeros_like(value["psi"]), atol=1e-11, rtol=0
        )
        torch.testing.assert_close(
            value["delta"], torch.zeros_like(value["delta"]), atol=1e-11, rtol=0
        )
        assert bool((value["gamma"] > 0).all())


def test_new_psi_is_not_dynamical_dependence_over_two() -> None:
    system = _system()
    projection = _projection()[:1]
    psi = emergence_from_model(system, projection)["psi"]
    dd_over_two = dynamical_dependence(system, projection) / 2.0
    assert not torch.allclose(psi, dd_over_two, rtol=1e-5, atol=1e-7)


def test_full_past_matches_long_finite_history_reference() -> None:
    system = _system()
    projection = _projection()
    expected = _finite_history_emergence_reference(system, projection, 10)
    actual = emergence_from_model(system, projection, history="full")
    for name in ("psi", "delta", "gamma"):
        torch.testing.assert_close(
            actual[name][0], expected[name], rtol=2e-9, atol=1e-10
        )


def test_full_past_batch_matches_individual_models() -> None:
    first = _system()
    coefficients = torch.cat([first.coefficients, first.coefficients * 0.78], dim=0)
    covariance = first.innovation_covariance.expand(2, -1, -1).clone()
    system = build_var_system(coefficients, covariance)
    projections = torch.stack([_projection(), _projection() * 1.3])
    batched = emergence_from_model(system, projections, history="full")
    for index in range(2):
        single_system = build_var_system(coefficients[index], covariance[index])
        single = emergence_from_model(
            single_system, projections[index], history="full"
        )
        for name in ("psi", "delta", "gamma"):
            torch.testing.assert_close(
                batched[name][index], single[name][0], rtol=1e-10, atol=1e-11
            )


def test_full_past_preserves_float32_and_agrees_with_float64() -> None:
    value32 = emergence_from_model(
        _system(torch.float32), _projection(torch.float32), history="full"
    )
    value64 = emergence_from_model(
        _system(torch.float64), _projection(torch.float64), history="full"
    )
    for name in ("psi", "delta", "gamma"):
        assert value32[name].dtype == torch.float32
        torch.testing.assert_close(
            value32[name].double(), value64[name], rtol=4e-5, atol=4e-6
        )


def test_full_history_does_not_depend_on_lag_argument() -> None:
    system = _system()
    projection = _projection()
    one = emergence_from_model(system, projection, history="full", lag=1)
    five = emergence_from_model(system, projection, history="full", lag=5)
    for name in ("psi", "delta", "gamma"):
        torch.testing.assert_close(one[name], five[name])


def test_compute_all_can_select_full_past_emergence() -> None:
    system = _system()
    projection = _projection()
    config = ModelMeasureConfig(
        macro_projection=projection,
        emergence_history="full",
    )
    aggregate = compute_all_model_measures(system, config)["emergence"]
    direct = emergence_from_model(system, projection, history="full")
    for name in ("psi", "delta", "gamma"):
        torch.testing.assert_close(aggregate[name], direct[name])
