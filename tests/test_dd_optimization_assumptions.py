"""Whitening invariance tests for general-covariance DD optimization."""

import math

import torch

from complextorch import (
    InnovationsStateSpace,
    dynamical_dependence,
    optimise_dynamical_dependence_proxy,
    optimise_dynamical_dependence_spectral,
)
from complextorch.dd_optimization import orthonormalise_projection


def _equivalent_systems(dtype=torch.float64):
    """Return the same process in identity and nonidentity innovation coordinates."""
    a = torch.tensor(
        [[0.42, 0.08, 0.00], [0.00, 0.31, 0.06], [0.02, 0.00, 0.24]],
        dtype=dtype,
    )
    c_white = torch.tensor(
        [[1.0, 0.2, 0.0], [0.1, 0.8, 0.15], [0.0, -0.1, 0.9]],
        dtype=dtype,
    )
    k_white = torch.tensor(
        [[0.30, 0.04, 0.00], [0.02, 0.24, 0.03], [0.00, 0.05, 0.20]],
        dtype=dtype,
    )
    b = torch.tensor(
        [[1.20, 0.00, 0.00], [0.25, 0.90, 0.00], [0.10, -0.12, 1.10]],
        dtype=dtype,
    )
    identity = torch.eye(3, dtype=dtype)
    white = InnovationsStateSpace(a, c_white, k_white, identity)
    k_physical_t = torch.linalg.solve_triangular(
        b.transpose(-1, -2), k_white.transpose(-1, -2), upper=True
    )
    physical = InnovationsStateSpace(
        a,
        b @ c_white,
        k_physical_t.transpose(-1, -2),
        b @ b.transpose(-1, -2),
    )
    return white, physical, b


def _physical_projection(white_projection, factor):
    """Map a whitened projection through B^{-1} without explicit inversion."""
    physical_t = torch.linalg.solve_triangular(
        factor.transpose(-1, -2),
        white_projection.transpose(-1, -2),
        upper=True,
    )
    return orthonormalise_projection(physical_t.transpose(-1, -2))


def _projector(projection):
    return projection.transpose(-1, -2) @ projection


def test_exact_dd_is_invariant_under_innovation_whitening():
    """Whitening and the matching projection map preserve exact DD."""
    white, physical, factor = _equivalent_systems()
    projections = [
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[0.4, 0.8, -0.2]], dtype=torch.float64),
        torch.tensor(
            [[1.0, 0.0, 0.2], [0.0, 1.0, -0.1]], dtype=torch.float64
        ),
    ]
    for white_projection in projections:
        white_projection = orthonormalise_projection(white_projection)
        physical_projection = _physical_projection(white_projection, factor)
        dd_white = dynamical_dependence(white, white_projection, base=math.e)
        dd_physical = dynamical_dependence(
            physical, physical_projection, base=math.e
        )
        torch.testing.assert_close(dd_physical, dd_white, rtol=2e-9, atol=2e-11)


def test_proxy_optimizer_matches_equivalent_whitened_problem():
    """General-V proxy optimization equals the identity-V problem after whitening."""
    white, physical, factor = _equivalent_systems()
    white_initial = orthonormalise_projection(
        torch.tensor([[0.6, -0.3, 0.7]], dtype=torch.float64)
    )
    physical_initial = _physical_projection(white_initial, factor)
    kwargs = dict(
        max_iterations=18,
        variant=1,
        initial_step_size=2e-3,
        tol=(1e-14, 1e-14, 1e-14),
        history=True,
    )
    expected = optimise_dynamical_dependence_proxy(white, white_initial, **kwargs)
    actual = optimise_dynamical_dependence_proxy(physical, physical_initial, **kwargs)

    torch.testing.assert_close(actual.objective, expected.objective, rtol=2e-10, atol=2e-12)
    torch.testing.assert_close(actual.step_size, expected.step_size, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        actual.history, expected.history, rtol=2e-10, atol=2e-12, equal_nan=True
    )
    assert torch.equal(actual.convergence, expected.convergence)
    assert torch.equal(actual.iterations, expected.iterations)

    expected_physical = _physical_projection(expected.projection, factor)
    torch.testing.assert_close(
        _projector(actual.projection[0]),
        _projector(expected_physical[0]),
        rtol=2e-9,
        atol=2e-11,
    )
    torch.testing.assert_close(
        dynamical_dependence(physical, actual.projection[0], base=math.e),
        dynamical_dependence(white, expected.projection[0], base=math.e),
        rtol=2e-9,
        atol=2e-11,
    )


def test_spectral_optimizer_matches_equivalent_whitened_problem():
    """General-V spectral optimization equals the whitened ComplexBox problem."""
    white, physical, factor = _equivalent_systems()
    white_initial = orthonormalise_projection(
        torch.tensor([[0.5, 0.4, -0.7]], dtype=torch.float64)
    )
    physical_initial = _physical_projection(white_initial, factor)
    frequencies = torch.linspace(0.0, 0.5, 33, dtype=torch.float64)
    kwargs = dict(
        max_iterations=14,
        variant=1,
        initial_step_size=1e-3,
        tol=(1e-14, 1e-14, 1e-14),
        history=True,
    )
    expected = optimise_dynamical_dependence_spectral(
        white, white_initial, frequencies, **kwargs
    )
    actual = optimise_dynamical_dependence_spectral(
        physical, physical_initial, frequencies, **kwargs
    )

    torch.testing.assert_close(actual.objective, expected.objective, rtol=3e-10, atol=3e-12)
    torch.testing.assert_close(
        actual.history, expected.history, rtol=3e-10, atol=3e-12, equal_nan=True
    )
    assert torch.equal(actual.convergence, expected.convergence)
    assert torch.equal(actual.iterations, expected.iterations)

    expected_physical = _physical_projection(expected.projection, factor)
    torch.testing.assert_close(
        _projector(actual.projection[0]),
        _projector(expected_physical[0]),
        rtol=3e-9,
        atol=3e-11,
    )


def test_general_covariance_optimizer_preserves_dtype_and_row_orthonormality():
    """Whitening stays Torch-native and returns a physical Stiefel basis."""
    _white, physical, factor = _equivalent_systems(dtype=torch.float32)
    initial_white = orthonormalise_projection(
        torch.tensor([[0.2, 0.9, -0.3]], dtype=torch.float32)
    )
    initial_physical = _physical_projection(initial_white, factor)
    result = optimise_dynamical_dependence_proxy(
        physical, initial_physical, max_iterations=3
    )
    assert result.projection.dtype == torch.float32
    identity = torch.eye(result.projection.shape[-2], dtype=torch.float32)
    torch.testing.assert_close(
        result.projection[0] @ result.projection[0].transpose(-1, -2),
        identity,
        rtol=2e-5,
        atol=2e-6,
    )
