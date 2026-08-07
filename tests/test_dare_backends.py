"""Parity and numerical-validation tests for DARE backends."""

import numpy as np
import pytest
import torch

import complextorch.control as control
from complextorch import solve_dare


def _fixture(dtype=torch.float64, spectral_radius=0.92):
    """Return a deterministic observable state-space DARE fixture."""
    transition = torch.tensor(
        [[0.78, 0.18, -0.04], [0.00, 0.62, 0.13], [0.05, 0.00, 0.55]],
        dtype=dtype,
    )
    current = torch.linalg.eigvals(transition).abs().amax().real
    transition = transition * (spectral_radius / current)
    observation = torch.tensor(
        [[1.0, 0.25, -0.10], [0.15, 0.80, 0.35]], dtype=dtype
    )
    process_covariance = torch.tensor(
        [[0.50, 0.08, 0.03], [0.08, 0.35, 0.04], [0.03, 0.04, 0.28]],
        dtype=dtype,
    )
    observation_covariance = torch.tensor(
        [[0.70, 0.12], [0.12, 0.55]], dtype=dtype
    )
    return transition, observation, process_covariance, observation_covariance


def _relative_residual(a, c, q, r, p):
    """Return the relative filtering-DARE residual."""
    innovation = c @ p @ c.transpose(-1, -2) + r
    numerator = a @ p @ c.transpose(-1, -2)
    rhs = (
        a @ p @ a.transpose(-1, -2)
        + q
        - numerator
        @ torch.linalg.solve(innovation, numerator.transpose(-1, -2))
    )
    residual = torch.linalg.matrix_norm(rhs - p, ord="fro", dim=(-2, -1))
    scale = torch.linalg.matrix_norm(p, ord="fro", dim=(-2, -1)).clamp_min(1.0)
    return residual / scale


def test_default_backend_remains_scipy_reference():
    """Adding backend selection must not change legacy solve_dare calls."""
    args = _fixture()
    implicit = solve_dare(*args)
    explicit = solve_dare(*args, backend="scipy")
    torch.testing.assert_close(implicit, explicit, rtol=0.0, atol=0.0)


def test_torch_dare_matches_scipy_float64():
    """The device-native solver must reproduce the SciPy reference solution."""
    args = _fixture(torch.float64)
    reference = solve_dare(*args, backend="scipy")
    actual = solve_dare(*args, backend="torch")
    torch.testing.assert_close(actual, reference, rtol=2e-10, atol=2e-12)
    assert float(_relative_residual(*args, actual)) < 2e-10


@pytest.mark.parametrize("spectral_radius", [0.5, 0.95, 0.999, 1.01, 1.20])
def test_torch_dare_parity_across_stability_boundary(spectral_radius):
    """SDA parity must hold near the unit circle and for detectable unstable A."""
    args = _fixture(torch.float64, spectral_radius=spectral_radius)
    reference = solve_dare(*args, backend="scipy")
    actual = solve_dare(*args, backend="torch")
    torch.testing.assert_close(actual, reference, rtol=2e-9, atol=2e-11)
    assert float(_relative_residual(*args, actual)) < 2e-9


def test_torch_dare_matches_scipy_float32_with_dtype_appropriate_tolerance():
    """Float32 output preserves dtype while using robust working precision."""
    args = _fixture(torch.float32)
    reference = solve_dare(*args, backend="scipy")
    actual = solve_dare(*args, backend="torch")
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, reference, rtol=2e-6, atol=2e-6)
    assert float(_relative_residual(*args, actual)) < 5e-5


def test_torch_dare_float32_stress_regression_uses_stable_working_precision():
    """A poorly conditioned float32 case must remain as accurate as the oracle."""
    a = torch.tensor(
        [
            [-0.14602912, -1.09824765, 0.05948520, -0.33553204],
            [-0.42419666, -0.25216226, 0.62077818, -0.02102504],
            [-0.25999812, 0.75788185, 0.75245970, -1.31482141],
            [0.29065972, -0.63579088, 0.49295250, 0.81676400],
        ],
        dtype=torch.float32,
    )
    c = torch.tensor(
        [[0.87648548, 1.30624304, -0.30995763, -0.76437592]],
        dtype=torch.float32,
    )
    q = torch.tensor(
        [
            [1.48888428, 0.56686883, 0.07339499, 0.61824252],
            [0.56686883, 3.02545708, -0.48017858, 0.55945247],
            [0.07339499, -0.48017858, 0.98240232, -0.40635106],
            [0.61824252, 0.55945247, -0.40635106, 5.44754635],
        ],
        dtype=torch.float32,
    )
    r = torch.tensor([[1.59525655]], dtype=torch.float32)

    reference = solve_dare(a, c, q, r, backend="scipy")
    actual = solve_dare(a, c, q, r, backend="torch")
    torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-5)

    reference_residual = float(_relative_residual(a, c, q, r, reference))
    actual_residual = float(_relative_residual(a, c, q, r, actual))
    assert actual_residual <= 2.0 * reference_residual + 1e-7


def test_torch_dare_is_batched_and_matches_individual_scipy_solves():
    """Batch execution must equal independent reference solves without Python batching."""
    first = _fixture(torch.float64, spectral_radius=0.75)
    second = _fixture(torch.float64, spectral_radius=0.995)
    a = torch.stack((first[0], second[0]))
    c = torch.stack((first[1], second[1]))
    q = torch.stack((first[2], 1.4 * second[2]))
    r = torch.stack((first[3], 0.8 * second[3]))

    actual = solve_dare(a, c, q, r, backend="torch")
    reference = torch.stack(
        [
            solve_dare(a[index], c[index], q[index], r[index], backend="scipy")
            for index in range(2)
        ]
    )
    assert actual.shape == (2, 3, 3)
    torch.testing.assert_close(actual, reference, rtol=2e-9, atol=2e-11)


def test_torch_dare_broadcasts_unbatched_model_components():
    """A singleton model component may be shared across a batched DARE input."""
    a, c, q, r = _fixture(torch.float64)
    q_batch = torch.stack((q, 1.3 * q, 0.7 * q))
    actual = solve_dare(a, c, q_batch, r, backend="torch")
    reference = torch.stack(
        [solve_dare(a, c, q_i, r, backend="scipy") for q_i in q_batch]
    )
    assert actual.shape == (3, 3, 3)
    torch.testing.assert_close(actual, reference, rtol=2e-9, atol=2e-11)


def test_torch_backend_does_not_call_scipy(monkeypatch):
    """Torch execution must not leave the Torch numerical path."""
    args = _fixture(torch.float64)

    def _forbidden(*args, **kwargs):
        raise AssertionError("SciPy backend was called")

    monkeypatch.setattr(control, "solve_discrete_are", _forbidden)
    result = solve_dare(*args, backend="torch")
    assert torch.isfinite(result).all()


def test_torch_dare_preserves_device():
    """The Torch backend must return on the input device."""
    args = _fixture(torch.float64)
    result = solve_dare(*args, backend="torch")
    assert result.device == args[0].device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_torch_dare_runs_natively_on_cuda():
    """CUDA input must remain on CUDA throughout the public Torch backend."""
    args = tuple(value.cuda() for value in _fixture(torch.float64))
    result = solve_dare(*args, backend="torch")
    assert result.is_cuda
    reference = solve_dare(*args, backend="scipy")
    torch.testing.assert_close(result, reference, rtol=2e-9, atol=2e-11)


def test_torch_dare_randomized_parity_grid():
    """Fixed-seed random systems provide regression coverage beyond one fixture."""
    rng = np.random.default_rng(20260807)
    worst_relative_error = 0.0
    worst_residual = 0.0

    for n_states, n_observations, radius in (
        (1, 1, 0.20),
        (2, 1, 0.90),
        (3, 2, 0.99),
        (4, 3, 1.05),
        (6, 4, 1.20),
    ):
        raw_a = rng.standard_normal((n_states, n_states))
        eig_radius = np.max(np.abs(np.linalg.eigvals(raw_a)))
        a = torch.tensor(raw_a * (radius / eig_radius), dtype=torch.float64)
        c = torch.tensor(
            rng.standard_normal((n_observations, n_states)), dtype=torch.float64
        )
        q_factor = rng.standard_normal((n_states, n_states))
        r_factor = rng.standard_normal((n_observations, n_observations))
        q = torch.tensor(
            q_factor @ q_factor.T + 0.2 * np.eye(n_states), dtype=torch.float64
        )
        r = torch.tensor(
            r_factor @ r_factor.T + 0.2 * np.eye(n_observations),
            dtype=torch.float64,
        )

        reference = solve_dare(a, c, q, r, backend="scipy")
        actual = solve_dare(a, c, q, r, backend="torch")
        relative_error = float(
            torch.linalg.matrix_norm(actual - reference, ord="fro")
            / torch.linalg.matrix_norm(reference, ord="fro").clamp_min(1.0)
        )
        residual = float(_relative_residual(a, c, q, r, actual))
        worst_relative_error = max(worst_relative_error, relative_error)
        worst_residual = max(worst_residual, residual)

    assert worst_relative_error < 5e-9
    assert worst_residual < 5e-9


def test_dare_rejects_unknown_backend():
    """Backend selection is explicit and closed over the supported implementations."""
    with pytest.raises(ValueError, match="backend"):
        solve_dare(*_fixture(), backend="unknown")


def test_torch_dare_rejects_incompatible_batch_dimensions():
    """Different non-singleton batch sizes must not be silently combined."""
    a, c, q, r = _fixture()
    with pytest.raises(ValueError, match="batch"):
        solve_dare(
            a.repeat(2, 1, 1),
            c.repeat(3, 1, 1),
            q,
            r,
            backend="torch",
        )


def test_torch_dare_rejects_mixed_dtypes():
    """Torch matrix algebra requires one explicit common floating dtype."""
    a, c, q, r = _fixture(torch.float64)
    with pytest.raises(ValueError, match="dtype"):
        solve_dare(a, c.float(), q, r, backend="torch")
