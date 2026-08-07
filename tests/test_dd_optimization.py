"""Parity tests for the initial ComplexBox/SSDI optimisation port."""

import numpy as np
import pytest
import torch

from complextorch.control import InnovationsStateSpace, innovations_transfer_function
from complextorch.dd_optimization import (
    innovations_proxy_sequence,
    optimise_dynamical_dependence_proxy,
    optimise_dynamical_dependence_spectral,
    orthonormalise_projection,
    proxy_dynamical_dependence,
    proxy_dynamical_dependence_gradient,
    spectral_dynamical_dependence,
    spectral_dynamical_dependence_gradient,
)


def _system(dtype=torch.float64, *, identity_covariance=True):
    covariance = (
        torch.eye(3, dtype=dtype)
        if identity_covariance
        else torch.tensor(
            [[1.0, 0.1, 0.0], [0.1, 0.8, -0.05], [0.0, -0.05, 1.2]],
            dtype=dtype,
        )
    )
    return InnovationsStateSpace(
        transition=torch.tensor(
            [[0.42, 0.08, 0.00], [0.00, 0.31, 0.05], [0.02, 0.00, 0.23]],
            dtype=dtype,
        ),
        observation=torch.tensor(
            [[1.00, 0.15, 0.00], [0.10, 0.85, 0.20], [0.00, -0.10, 0.90]],
            dtype=dtype,
        ),
        gain=torch.tensor(
            [[0.34, 0.05, 0.01], [0.03, 0.27, 0.04], [0.00, 0.02, 0.22]],
            dtype=dtype,
        ),
        innovation_covariance=covariance,
    )


def _numpy_orthonormalise_rows(matrix):
    """ComplexBox orthonormalise(L).T for M=L.T."""
    _u, _s, vh = np.linalg.svd(np.asarray(matrix), full_matrices=False)
    return vh


def _initial_projections():
    raw = np.array(
        [
            [[1.0, 0.2, -0.3], [0.1, 0.9, 0.4]],
            [[0.3, 1.0, 0.2], [0.8, -0.1, 0.5]],
            [[0.7, -0.4, 0.8], [0.2, 0.9, -0.1]],
        ],
        dtype=float,
    )
    return np.stack([_numpy_orthonormalise_rows(value) for value in raw])


def _numpy_proxy_value(matrix, sequence):
    l = matrix.T
    value = 0.0
    for q in sequence:
        left = l.T @ q
        value += np.sum(left * left) - np.sum((left @ l) ** 2)
    return float(value)


def _numpy_proxy_gradient(matrix, sequence):
    l = matrix.T
    p = l @ l.T
    g = np.zeros((l.shape[0], l.shape[0]))
    for q in sequence:
        g += q @ q.T - q.T @ p @ q - q @ p @ q.T
    grad = 2.0 * g @ l
    grad = grad - p @ grad
    return grad.T, float(np.linalg.norm(grad))


def _numpy_spectral_value(matrix, transfer):
    l = matrix.T
    values = []
    for h in transfer:
        hl = h.conj().T @ l
        gram = hl.conj().T @ hl
        chol = np.linalg.cholesky(gram)
        values.append(2.0 * np.sum(np.log(np.real(np.diag(chol)))))
    values = np.asarray(values)
    weights = np.ones(values.size)
    weights[[0, -1]] = 0.5
    return float(np.sum(values * weights) / (values.size - 1))


def _numpy_spectral_gradient(matrix, transfer):
    l = matrix.T
    weights = np.ones(len(transfer))
    weights[[0, -1]] = 0.5
    weights /= len(transfer) - 1
    grad = np.zeros_like(l)
    for weight, h in zip(weights, transfer, strict=True):
        hl = h.conj().T @ l
        gram = hl.conj().T @ hl
        grad += 2.0 * weight * np.real((h @ hl) @ np.linalg.inv(gram))
    grad -= 2.0 * l
    return grad.T, float(np.linalg.norm(grad))


def _numpy_proxy_optimise(initial, sequence, *, variant, max_iterations=35):
    """Independent direct port of ComplexBox opt_gd{1,2}_ddx_mruns."""
    dds, matrices, conv, sigma, iterations, histories = [], [], [], [], [], []
    for start in initial:
        matrix = start.copy()
        grad, gmag = _numpy_proxy_gradient(matrix, sequence)
        dd = _numpy_proxy_value(matrix, sequence)
        step = 1e-3
        history = [(dd, step, gmag)]
        code = 0
        stop = max_iterations
        for iteration in range(2, max_iterations + 1):
            candidate = _numpy_orthonormalise_rows(
                matrix - step * (grad / gmag) if gmag > 0 else matrix
            )
            dd_try = _numpy_proxy_value(candidate, sequence)
            if variant == 1:
                if dd_try < dd:
                    matrix = candidate
                    grad, gmag = _numpy_proxy_gradient(matrix, sequence)
                    dd = dd_try
                    step *= 2.0
                else:
                    step *= 0.5
            else:
                matrix = candidate
                grad, gmag = _numpy_proxy_gradient(matrix, sequence)
                dd_new = _numpy_proxy_value(matrix, sequence)
                if dd_new < dd:
                    dd = dd_new
                    step *= 2.0
                else:
                    step *= 0.5
            history.append((dd, step, gmag))
            if step < 1e-9:
                code, stop = 1, iteration
                break
            if dd < 1e-9:
                code, stop = 2, iteration
                break
            if gmag < 1e-10:
                code, stop = 3, iteration
                break
        dds.append(dd)
        matrices.append(matrix)
        conv.append(code)
        sigma.append(step)
        iterations.append(stop)
        histories.append(np.asarray(history))
    order = np.argsort(dds, kind="stable")
    return (
        np.asarray(dds)[order],
        np.asarray(matrices)[order],
        np.asarray(conv)[order],
        np.asarray(sigma)[order],
        np.asarray(iterations)[order],
        [histories[index] for index in order],
    )


def test_proxy_sequence_matches_complexbox_iss2cak_equation():
    system = _system()
    actual = innovations_proxy_sequence(system, lags=5).numpy()
    a = system.transition.numpy()
    c = system.observation.numpy()
    k = system.gain.numpy()
    expected = []
    power = np.eye(3)
    for lag in range(5):
        if lag:
            power = power @ a
        expected.append(c @ power @ k)
    np.testing.assert_allclose(actual, np.stack(expected), rtol=1e-13, atol=1e-14)


def test_proxy_objective_and_gradient_match_complexbox_equations():
    system = _system()
    sequence = innovations_proxy_sequence(system, lags=4)
    projections = torch.tensor(_initial_projections(), dtype=torch.float64)
    actual = proxy_dynamical_dependence(projections, sequence)
    grad, magnitude = proxy_dynamical_dependence_gradient(projections, sequence)
    expected = np.array(
        [_numpy_proxy_value(value, sequence.numpy()) for value in projections.numpy()]
    )
    expected_grad = np.stack(
        [_numpy_proxy_gradient(value, sequence.numpy())[0] for value in projections.numpy()]
    )
    expected_mag = np.array(
        [_numpy_proxy_gradient(value, sequence.numpy())[1] for value in projections.numpy()]
    )
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(grad.numpy(), expected_grad, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(magnitude.numpy(), expected_mag, rtol=1e-11, atol=1e-12)
    tangent = grad @ projections.transpose(-1, -2)
    torch.testing.assert_close(tangent, torch.zeros_like(tangent), rtol=0.0, atol=1e-11)


@pytest.mark.parametrize("variant", [1, 2])
def test_proxy_optimizer_matches_independent_complexbox_loop(variant):
    system = _system()
    sequence = innovations_proxy_sequence(system, lags=4).numpy()
    initial = _initial_projections()
    expected = _numpy_proxy_optimise(initial, sequence, variant=variant)
    actual = optimise_dynamical_dependence_proxy(
        system,
        torch.tensor(initial, dtype=torch.float64),
        lags=4,
        max_iterations=35,
        variant=variant,
        history=True,
    )
    np.testing.assert_allclose(actual.objective.numpy(), expected[0], rtol=1e-9, atol=1e-11)
    np.testing.assert_array_equal(actual.convergence.numpy(), expected[2])
    np.testing.assert_allclose(actual.step_size.numpy(), expected[3], rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(actual.iterations.numpy(), expected[4])
    actual_projector = actual.projection.transpose(-1, -2) @ actual.projection
    expected_projector = np.stack([value.T @ value for value in expected[1]])
    np.testing.assert_allclose(actual_projector.numpy(), expected_projector, rtol=1e-8, atol=1e-10)
    for run, reference_history in enumerate(expected[5]):
        torch_history = actual.history[run, : len(reference_history)].numpy()
        np.testing.assert_allclose(torch_history, reference_history, rtol=1e-8, atol=1e-11)


def test_spectral_objective_and_gradient_match_complexbox_equations():
    system = _system()
    frequencies = torch.linspace(0.0, 0.5, 17, dtype=torch.float64)
    transfer = innovations_transfer_function(system, frequencies)
    projections = torch.tensor(_initial_projections(), dtype=torch.float64)
    actual = spectral_dynamical_dependence(projections, transfer)
    grad, magnitude = spectral_dynamical_dependence_gradient(projections, transfer)
    transfer_np = transfer.numpy()
    expected = np.array(
        [_numpy_spectral_value(value, transfer_np) for value in projections.numpy()]
    )
    expected_grad = np.stack(
        [_numpy_spectral_gradient(value, transfer_np)[0] for value in projections.numpy()]
    )
    expected_mag = np.array(
        [_numpy_spectral_gradient(value, transfer_np)[1] for value in projections.numpy()]
    )
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(grad.numpy(), expected_grad, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(magnitude.numpy(), expected_mag, rtol=1e-9, atol=1e-10)


def test_spectral_optimizer_supports_identity_and_general_covariance():
    initial = torch.tensor(_initial_projections(), dtype=torch.float64)
    frequencies = torch.linspace(0.0, 0.5, 9, dtype=torch.float64)
    for system in (_system(), _system(identity_covariance=False)):
        result = optimise_dynamical_dependence_spectral(
            system,
            initial,
            frequencies,
            max_iterations=8,
            variant=2,
        )
        assert result.objective.shape == (3,)
        assert result.projection.shape == (3, 2, 3)
        assert torch.all(result.objective[1:] >= result.objective[:-1])


def test_optimizers_preserve_float32_dtype_and_device():
    system = _system(torch.float32)
    initial = torch.tensor(_initial_projections(), dtype=torch.float32)
    result = optimise_dynamical_dependence_proxy(
        system, initial, lags=3, max_iterations=4
    )
    assert result.objective.dtype == torch.float32
    assert result.projection.dtype == torch.float32
    assert result.objective.device == system.transition.device
    assert result.projection.device == system.transition.device


def test_orthonormalise_projection_matches_row_stiefel_contract():
    matrix = torch.tensor(
        [[1.0, 2.0, -0.3], [0.2, -0.1, 0.7]], dtype=torch.float64
    )
    orthogonal = orthonormalise_projection(matrix)
    torch.testing.assert_close(
        orthogonal @ orthogonal.T,
        torch.eye(2, dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )
