import math

import pytest
import torch

from complextorch import (
    InnovationsStateSpace,
    dynamical_dependence,
    innovations_proxy_sequence,
    innovations_transfer_function,
    optimise_dynamical_dependence_proxy,
    optimise_dynamical_dependence_proxy_riemannian,
    optimise_dynamical_dependence_spectral_riemannian,
    orthonormalise_projection,
    proxy_dynamical_dependence_autograd_gradient,
    proxy_dynamical_dependence_gradient,
    spectral_dynamical_dependence_autograd_gradient,
    spectral_dynamical_dependence_gradient,
)


def _white_system(dtype=torch.float64, device="cpu"):
    a = torch.tensor(
        [
            [0.42, 0.08, 0.00],
            [0.00, 0.31, 0.06],
            [0.02, 0.00, 0.24],
        ],
        dtype=dtype,
        device=device,
    )
    c = torch.tensor(
        [
            [1.0, 0.2, 0.0],
            [0.1, 0.8, 0.15],
            [0.0, -0.1, 0.9],
        ],
        dtype=dtype,
        device=device,
    )
    k = torch.tensor(
        [
            [0.30, 0.04, 0.00],
            [0.02, 0.24, 0.03],
            [0.00, 0.05, 0.20],
        ],
        dtype=dtype,
        device=device,
    )
    return InnovationsStateSpace(
        a,
        c,
        k,
        torch.eye(3, dtype=dtype, device=device),
    )


def _equivalent_physical_system(dtype=torch.float64, device="cpu"):
    white = _white_system(dtype=dtype, device=device)
    factor = torch.tensor(
        [
            [1.20, 0.00, 0.00],
            [0.25, 0.90, 0.00],
            [0.10, -0.12, 1.10],
        ],
        dtype=dtype,
        device=device,
    )
    physical_gain_t = torch.linalg.solve_triangular(
        factor.transpose(-1, -2),
        white.gain.transpose(-1, -2),
        upper=True,
    )
    physical = InnovationsStateSpace(
        white.transition,
        factor @ white.observation,
        physical_gain_t.transpose(-1, -2),
        factor @ factor.transpose(-1, -2),
    )
    return white, physical, factor


def _physical_projection(white_projection, factor):
    physical_t = torch.linalg.solve_triangular(
        factor.transpose(-1, -2),
        white_projection.transpose(-1, -2),
        upper=True,
    )
    return orthonormalise_projection(physical_t.transpose(-1, -2))


def _projector(matrix):
    return matrix.transpose(-1, -2) @ matrix


def test_proxy_autograd_oracle_matches_analytic_grassmann_gradient():
    system = _white_system()
    sequence = innovations_proxy_sequence(system)
    projection = orthonormalise_projection(
        torch.tensor(
            [[0.6, -0.3, 0.7], [0.1, 0.9, 0.2]], dtype=torch.float64
        )
    )
    analytic, analytic_norm = proxy_dynamical_dependence_gradient(
        projection, sequence
    )
    oracle, oracle_norm = proxy_dynamical_dependence_autograd_gradient(
        projection, sequence
    )
    torch.testing.assert_close(oracle, analytic, rtol=2e-10, atol=2e-12)
    torch.testing.assert_close(oracle_norm, analytic_norm, rtol=2e-10, atol=2e-12)


def test_spectral_autograd_oracle_matches_analytic_grassmann_gradient():
    system = _white_system()
    frequencies = torch.linspace(0.0, 0.5, 33, dtype=torch.float64)
    transfer = innovations_transfer_function(system, frequencies)
    projection = orthonormalise_projection(
        torch.tensor([[0.5, 0.4, -0.7]], dtype=torch.float64)
    )
    analytic, analytic_norm = spectral_dynamical_dependence_gradient(
        projection, transfer
    )
    oracle, oracle_norm = spectral_dynamical_dependence_autograd_gradient(
        projection, transfer
    )
    torch.testing.assert_close(oracle, analytic, rtol=2e-9, atol=2e-11)
    torch.testing.assert_close(oracle_norm, analytic_norm, rtol=2e-9, atol=2e-11)


def test_proxy_armijo_is_monotone_and_counts_rejected_evaluations():
    system = _white_system()
    initial = orthonormalise_projection(
        torch.tensor([[0.6, -0.3, 0.7]], dtype=torch.float64)
    )
    result = optimise_dynamical_dependence_proxy_riemannian(
        system,
        initial,
        max_iterations=40,
        initial_step_size=2.0,
        history=True,
    )
    valid = result.history[0, :, 0]
    valid = valid[torch.isfinite(valid)]
    assert valid.numel() >= 2
    assert bool(torch.all(valid[1:] <= valid[:-1] + 1e-12))
    assert int(result.objective_evaluations[0]) >= int(result.iterations[0])
    assert int(result.gradient_evaluations[0]) == int(result.iterations[0])
    assert int(result.backtracking_evaluations[0]) == (
        int(result.objective_evaluations[0]) - int(result.iterations[0])
    )
    identity = torch.eye(result.projection.shape[-2], dtype=torch.float64)
    torch.testing.assert_close(
        result.projection[0] @ result.projection[0].T,
        identity,
        rtol=1e-10,
        atol=1e-12,
    )


def test_riemannian_proxy_general_covariance_matches_whitened_problem():
    white, physical, factor = _equivalent_physical_system()
    white_initial = orthonormalise_projection(
        torch.tensor([[0.6, -0.3, 0.7]], dtype=torch.float64)
    )
    physical_initial = _physical_projection(white_initial, factor)
    kwargs = dict(
        max_iterations=35,
        initial_step_size=1.0,
        gradient_tolerance=1e-11,
        objective_tolerance=1e-14,
    )
    expected = optimise_dynamical_dependence_proxy_riemannian(
        white, white_initial, **kwargs
    )
    actual = optimise_dynamical_dependence_proxy_riemannian(
        physical, physical_initial, **kwargs
    )
    torch.testing.assert_close(actual.objective, expected.objective, rtol=2e-9, atol=2e-11)
    expected_physical = _physical_projection(expected.projection[0], factor)
    torch.testing.assert_close(
        _projector(actual.projection[0]),
        _projector(expected_physical),
        rtol=2e-8,
        atol=2e-10,
    )
    torch.testing.assert_close(
        dynamical_dependence(physical, actual.projection[0], base=math.e),
        dynamical_dependence(white, expected.projection[0], base=math.e),
        rtol=2e-8,
        atol=2e-10,
    )


def test_riemannian_spectral_improves_objective_and_returns_finite_projection():
    system = _white_system()
    initial = orthonormalise_projection(
        torch.tensor([[0.5, 0.4, -0.7]], dtype=torch.float64)
    )
    frequencies = torch.linspace(0.0, 0.5, 33, dtype=torch.float64)
    result = optimise_dynamical_dependence_spectral_riemannian(
        system,
        initial,
        frequencies,
        max_iterations=25,
        initial_step_size=1.0,
        history=True,
    )
    first = result.history[0, 0, 0]
    assert bool(torch.isfinite(result.objective).all())
    assert bool(torch.isfinite(result.projection).all())
    assert bool(result.objective[0] <= first + 1e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_riemannian_proxy_preserves_dtype_and_is_numerically_finite(dtype):
    system = _white_system(dtype=dtype)
    initial = orthonormalise_projection(
        torch.tensor([[0.2, 0.9, -0.3]], dtype=dtype)
    )
    result = optimise_dynamical_dependence_proxy_riemannian(
        system,
        initial,
        max_iterations=8,
    )
    assert result.projection.dtype == dtype
    assert result.objective.dtype == dtype
    assert bool(torch.isfinite(result.projection).all())
    assert bool(torch.isfinite(result.objective).all())


def test_frozen_complexbox_baseline_remains_callable_and_separate():
    system = _white_system()
    initial = orthonormalise_projection(
        torch.tensor([[0.6, -0.3, 0.7]], dtype=torch.float64)
    )
    baseline = optimise_dynamical_dependence_proxy(
        system,
        initial,
        max_iterations=6,
        variant=1,
        initial_step_size=2e-3,
    )
    riemannian = optimise_dynamical_dependence_proxy_riemannian(
        system,
        initial,
        max_iterations=6,
        initial_step_size=1.0,
    )
    assert baseline.__class__.__name__ == "DDGradientSearchResult"
    assert riemannian.__class__.__name__ == "DDRiemannianSearchResult"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_riemannian_proxy_cuda():
    system = _white_system(device="cuda")
    initial = orthonormalise_projection(
        torch.tensor([[0.2, 0.9, -0.3]], dtype=torch.float64, device="cuda")
    )
    result = optimise_dynamical_dependence_proxy_riemannian(
        system,
        initial,
        max_iterations=8,
    )
    assert result.projection.is_cuda
    assert result.objective.is_cuda
    assert bool(torch.isfinite(result.objective).all())
