import inspect

import torch

from complextorch import optimise_dynamical_dependence
from complextorch.dd_ssdi import grassmann_distance_matrix, lcluster
from complextorch.dd_optimization import (
    orthonormalise_projection,
    proxy_dynamical_dependence,
)
from complextorch.representations import build_var_system


def _var(dtype=torch.float64):
    coefficients = torch.tensor(
        [
            [
                [0.22, 0.03, 0.00],
                [0.00, 0.18, 0.02],
                [0.01, 0.00, 0.16],
            ],
            [
                [0.04, 0.00, 0.00],
                [0.01, 0.03, 0.00],
                [0.00, 0.01, 0.02],
            ],
        ],
        dtype=dtype,
    )
    factor = torch.tensor(
        [[1.2, 0.0, 0.0], [0.2, 0.9, 0.0], [0.1, -0.05, 1.1]],
        dtype=dtype,
    )
    return build_var_system(coefficients, factor @ factor.T)


def _initial(dtype=torch.float64):
    raw = torch.tensor(
        [
            [[0.8, 0.2, -0.1]],
            [[0.1, 0.9, 0.3]],
            [[-0.4, 0.3, 0.8]],
            [[0.5, -0.7, 0.2]],
        ],
        dtype=dtype,
    )
    return orthonormalise_projection(raw)


def test_default_public_dd_workflow_is_staged_ssdi():
    signature = inspect.signature(optimise_dynamical_dependence)
    assert signature.parameters["objective"].default is None
    assert signature.parameters["optimizer"].default == "complexbox"
    assert signature.parameters["restarts"].default == 100
    assert signature.parameters["cluster_tolerance"].default == 1e-6

    result = optimise_dynamical_dependence(
        _var(),
        _initial(),
        max_iterations=2,
        frequency_bins=9,
        cluster_tolerance=1e-12,
    )
    assert result.workflow == "ssdi"
    assert result.objective_name == "spectral"
    assert result.preoptimization is not None
    assert result.preoptimization.objective_name == "proxy"
    assert result.cluster_indices is not None
    assert result.cluster_sizes is not None
    assert result.cluster_distances is not None
    assert int(result.cluster_sizes.sum()) == 4


def test_staged_var_proxy_uses_transformed_var_coefficients_directly():
    system = _var()
    initial = _initial()
    result = optimise_dynamical_dependence(
        system,
        initial,
        max_iterations=1,
        frequencies=torch.linspace(0.0, 0.5, 9, dtype=torch.float64),
        cluster_tolerance=1e-12,
    )
    assert result.preoptimization is not None

    covariance = system.innovation_covariance[0]
    factor = torch.linalg.cholesky(covariance)
    coefficients = system.coefficients[0]
    transformed = torch.linalg.solve_triangular(
        factor.expand(coefficients.shape[0], -1, -1),
        coefficients @ factor,
        upper=False,
    )
    transformed_initial = orthonormalise_projection(initial @ factor)
    expected = torch.sort(
        proxy_dynamical_dependence(transformed_initial, transformed)
    ).values
    torch.testing.assert_close(
        result.preoptimization.objective,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_staged_random_initialization_is_seeded_and_batched():
    kwargs = dict(
        output_dimension=1,
        restarts=6,
        seed=31415,
        max_iterations=2,
        frequency_bins=9,
        cluster_tolerance=1e-8,
    )
    first = optimise_dynamical_dependence(_var(), **kwargs)
    second = optimise_dynamical_dependence(_var(), **kwargs)
    assert first.preoptimization is not None
    assert second.preoptimization is not None
    torch.testing.assert_close(
        first.preoptimization.projection,
        second.preoptimization.projection,
    )
    assert first.preoptimization.projection.shape == (6, 1, 3)
    assert int(first.cluster_sizes.sum()) == 6


def test_riemannian_backend_follows_same_staged_contract():
    result = optimise_dynamical_dependence(
        _var(),
        _initial(),
        optimizer="riemannian_armijo",
        max_iterations=2,
        frequency_bins=9,
        cluster_tolerance=1e-12,
    )
    assert result.workflow == "ssdi"
    assert result.optimizer == "riemannian_armijo"
    assert result.preoptimization is not None
    assert result.preoptimization.optimizer == "riemannian_armijo"
    assert result.objective_name == "spectral"


def test_lcluster_matches_sorted_greedy_ssdi_semantics():
    theta = torch.tensor([0.0, 0.001, 0.5, 0.501], dtype=torch.float64)
    bases = torch.stack(
        [torch.stack((torch.cos(t), torch.sin(t))).unsqueeze(0) for t in theta]
    )
    distance = grassmann_distance_matrix(bases)
    representatives, sizes = lcluster(distance, tolerance=0.01)
    assert representatives.tolist() == [0, 2]
    assert sizes.tolist() == [2, 2]
    torch.testing.assert_close(distance, distance.T, rtol=0, atol=1e-15)
    torch.testing.assert_close(
        torch.diagonal(distance), torch.zeros(4, dtype=distance.dtype)
    )


def _nested_var(dtype=torch.float64):
    from complextorch.representations import companion_matrix

    generator = torch.Generator(device="cpu").manual_seed(1234)
    n, order = 8, 4
    mask = torch.zeros((n, n), dtype=dtype)
    mask[:2, :2] = 1.0
    mask[2:6, :6] = 1.0
    mask[6:, :] = 1.0
    coefficients = torch.randn(
        (order, n, n), generator=generator, dtype=dtype
    ) * mask * 0.12
    for lag in range(order):
        coefficients[lag] *= 0.75**lag

    lo, hi = 0.0, 3.0
    for _ in range(60):
        scale = 0.5 * (lo + hi)
        companion = companion_matrix((coefficients * scale).unsqueeze(0))[0]
        radius = torch.linalg.eigvals(companion).abs().max().item()
        if radius < 0.82:
            lo = scale
        else:
            hi = scale
    coefficients = coefficients * lo

    noise_generator = torch.Generator(device="cpu").manual_seed(4321)
    raw = torch.randn((n, n), generator=noise_generator, dtype=dtype)
    covariance = raw @ raw.T / n + 0.5 * torch.eye(n, dtype=dtype)
    return build_var_system(coefficients, covariance)


def test_nested_var_recovers_only_the_planted_m2_and_m6_zero_dd_macros():
    system = _nested_var()
    best = {}
    for m in range(1, 8):
        result = optimise_dynamical_dependence(
            system,
            output_dimension=m,
            restarts=12,
            seed=900 + m,
            max_iterations=500,
            frequency_bins=33,
            cluster_tolerance=1e-3,
        )
        best[m] = float(result.objective.min())

    assert best[2] < 1e-5
    assert best[6] < 1e-5
    for m in (1, 3, 4, 5, 7):
        assert best[m] > 1e-3
