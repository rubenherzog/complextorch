"""Tests for the staged Barnett--Seth/SSDI optimization workflow."""
from __future__ import annotations

import torch

from complextorch import (
    DDSSDIOptimizationResult,
    build_var_system,
    optimise_dynamical_dependence,
    proxy_dynamical_dependence,
)
from complextorch.dd_optimization import _var_proxy_inputs, grassmann_distances, lcluster


def _small_var(dtype=torch.float64):
    coefficients = torch.zeros((2, 3, 3), dtype=dtype)
    coefficients[0] = torch.tensor(
        [[0.30, 0.00, 0.00], [0.10, 0.25, 0.00], [0.00, 0.10, 0.20]],
        dtype=dtype,
    )
    coefficients[1] = torch.tensor(
        [[0.05, 0.00, 0.00], [0.00, 0.04, 0.00], [0.00, 0.00, 0.03]],
        dtype=dtype,
    )
    covariance = torch.tensor(
        [[1.4, 0.2, 0.0], [0.2, 0.9, 0.1], [0.0, 0.1, 1.1]], dtype=dtype
    )
    return build_var_system(coefficients, covariance)


def test_lcluster_matches_matlab_greedy_semantics():
    distances = torch.tensor(
        [
            [0.0, 0.005, 0.2, 0.3],
            [0.005, 0.0, 0.2, 0.3],
            [0.2, 0.2, 0.0, 0.004],
            [0.3, 0.3, 0.004, 0.0],
        ],
        dtype=torch.float64,
    )
    dd = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    representatives, sizes = lcluster(distances, dd, tolerance=0.01)
    assert representatives.tolist() == [0, 2]
    assert sizes.tolist() == [2, 2]


def test_grassmann_distance_uses_normalized_max_principal_angle():
    projection = torch.tensor(
        [[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0]]],
        dtype=torch.float64,
    )
    distance = grassmann_distances(projection)
    torch.testing.assert_close(distance.diag(), torch.zeros(3, dtype=torch.float64))
    torch.testing.assert_close(distance[0, 1], torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(distance[0, 2], torch.tensor(0.0, dtype=torch.float64))


def test_var_proxy_inputs_match_matlab_residual_transform():
    system = _small_var()
    initial = torch.eye(3, dtype=torch.float64)[:1].unsqueeze(0)
    sequence, transformed, factor, identity = _var_proxy_inputs(
        system, initial, lags=None
    )
    assert not identity
    coefficients = system.coefficients[0]
    covariance = system.innovation_covariance[0]
    expected_factor = torch.linalg.cholesky(covariance)
    expected = torch.stack(
        [torch.linalg.solve(expected_factor, a @ expected_factor) for a in coefficients]
    )
    torch.testing.assert_close(factor, expected_factor)
    torch.testing.assert_close(sequence, expected)
    gram = transformed @ transformed.transpose(-1, -2)
    torch.testing.assert_close(gram, torch.ones_like(gram), atol=1e-12, rtol=1e-12)


def test_staged_ssdi_is_default_and_preserves_variant1_endpoint_objective():
    system = _small_var()
    result = optimise_dynamical_dependence(
        system,
        output_dimension=1,
        preoptimization_runs=6,
        preoptimization_max_iterations=40,
        spectral_max_iterations=30,
        frequency_points=17,
        random_seed=9,
    )
    assert isinstance(result, DDSSDIOptimizationResult)
    assert result.preoptimization.objective.shape == (6,)
    assert int(result.cluster_sizes.sum()) == 6
    assert result.objective.ndim == 1
    assert result.projection.shape[0] == result.cluster_sizes.numel()
    assert result.frequencies.shape == (17,)

    identity_system = build_var_system(
        system.coefficients[0], torch.eye(3, dtype=torch.float64)
    )
    initial = torch.randn((5, 1, 3), dtype=torch.float64)
    staged = optimise_dynamical_dependence(
        identity_system,
        initial,
        preoptimization_max_iterations=30,
        spectral_max_iterations=5,
        frequency_points=9,
    )
    reevaluated = proxy_dynamical_dependence(
        staged.preoptimization.projection, identity_system.coefficients[0]
    )
    torch.testing.assert_close(
        staged.preoptimization.objective, reevaluated, atol=1e-12, rtol=1e-10
    )


def test_staged_ssdi_riemannian_uses_same_workflow_contract():
    system = _small_var()
    initial = torch.randn((5, 1, 3), dtype=torch.float64)
    result = optimise_dynamical_dependence(
        system,
        initial,
        optimizer="riemannian_armijo",
        preoptimization_max_iterations=30,
        spectral_max_iterations=20,
        frequency_points=17,
    )
    assert isinstance(result, DDSSDIOptimizationResult)
    assert result.preoptimization.optimizer == "armijo"
    assert result.spectral.optimizer == "armijo"
    assert int(result.cluster_sizes.sum()) == 5


def test_explicit_objective_preserves_legacy_single_stage_api():
    system = _small_var()
    initial = torch.randn((3, 1, 3), dtype=torch.float64)
    result = optimise_dynamical_dependence(
        system,
        initial,
        objective="proxy",
        lags=2,
        max_iterations=5,
    )
    assert not isinstance(result, DDSSDIOptimizationResult)
    assert result.objective_name == "proxy"
    assert result.objective.shape == (3,)
