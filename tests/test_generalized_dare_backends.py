import numpy as np
import pytest
import torch
from scipy.linalg import solve_discrete_are

import complextorch.control as control
from complextorch.control import solve_dare, solve_generalized_dare


def _fixture(dtype=torch.float64):
    torch.manual_seed(20260807)
    n_states, n_observations = 4, 2
    a = torch.randn(n_states, n_states, dtype=dtype) * 0.18
    c = torch.randn(n_observations, n_states, dtype=dtype)

    # Build one positive-definite joint noise covariance so that R > 0 and
    # Q - S R^-1 S' > 0, the covariance conditions behind decorrelation.
    joint_dim = n_states + n_observations
    raw = torch.randn(joint_dim, joint_dim, dtype=dtype)
    joint = raw @ raw.T + 0.5 * torch.eye(joint_dim, dtype=dtype)
    q = joint[:n_states, :n_states]
    s = joint[:n_states, n_states:]
    r = joint[n_states:, n_states:]
    return a, c, q, r, s


def _scipy_reference(a, c, q, r, s):
    p = solve_discrete_are(
        a.detach().cpu().numpy().T,
        c.detach().cpu().numpy().T,
        q.detach().cpu().numpy(),
        r.detach().cpu().numpy(),
        s=s.detach().cpu().numpy(),
    )
    return torch.as_tensor(p, dtype=a.dtype, device=a.device)


def _relative_residual(a, c, q, r, s, p):
    innovation = c @ p @ c.T + r
    numerator = a @ p @ c.T + s
    correction = numerator @ torch.linalg.solve(innovation, numerator.T)
    rhs = a @ p @ a.T + q - correction
    return torch.linalg.matrix_norm(p - rhs) / torch.linalg.matrix_norm(p).clamp_min(1.0)


def test_default_preserves_existing_torch_backend_semantics():
    args = _fixture()
    default = solve_generalized_dare(*args)
    explicit = solve_generalized_dare(*args, backend="torch")
    assert torch.equal(default, explicit)


def test_torch_float64_matches_direct_scipy_generalized_dare():
    args = _fixture(torch.float64)
    actual = solve_generalized_dare(*args, backend="torch")
    expected = _scipy_reference(*args)
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-11)
    assert _relative_residual(*args, actual) < 1e-10


def test_torch_float32_matches_scipy_at_single_precision():
    args = _fixture(torch.float32)
    actual = solve_generalized_dare(*args, backend="torch")
    expected = _scipy_reference(*args)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert actual.dtype == torch.float32
    assert _relative_residual(*args, actual) < 2e-4


def test_scipy_backend_is_direct_complexbox_mdare_convention():
    # ComplexBox commit 87b5e2c maps its mdare(A,C,Q,R,S) directly to
    # scipy.linalg.solve_discrete_are(A.T, C.T, Q, R, s=S).
    args = _fixture(torch.float64)
    actual = solve_generalized_dare(*args, backend="scipy")
    expected = _scipy_reference(*args)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_zero_cross_covariance_reduces_to_standard_dare():
    a, c, q, r, s = _fixture(torch.float64)
    s = torch.zeros_like(s)
    generalized = solve_generalized_dare(a, c, q, r, s, backend="torch")
    standard = solve_dare(a, c, q, r, backend="torch")
    torch.testing.assert_close(generalized, standard, rtol=1e-11, atol=1e-12)


def test_batched_generalized_dare_matches_independent_reference_solves():
    args1 = _fixture(torch.float64)
    a, c, q, r, s = args1
    args2 = (
        a * 0.9,
        c * 1.1,
        q * 1.05,
        r * 0.95,
        s * 0.8,
    )
    stacked = tuple(torch.stack((x, y)) for x, y in zip(args1, args2, strict=True))
    actual = solve_generalized_dare(*stacked, backend="torch")
    expected = torch.stack((_scipy_reference(*args1), _scipy_reference(*args2)))
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-11)


def test_singleton_component_broadcasting_is_supported():
    a, c, q, r, s = _fixture(torch.float64)
    q_batch = torch.stack((q, q * 1.1))
    result = solve_generalized_dare(a, c, q_batch, r, s, backend="torch")
    expected = torch.stack(
        (
            _scipy_reference(a, c, q, r, s),
            _scipy_reference(a, c, q * 1.1, r, s),
        )
    )
    assert result.shape == (2, a.shape[0], a.shape[0])
    torch.testing.assert_close(result, expected, rtol=1e-9, atol=1e-11)


def test_torch_backend_never_calls_scipy(monkeypatch):
    args = _fixture(torch.float64)

    def _fail(*_args, **_kwargs):
        raise AssertionError("SciPy must not be called by the Torch backend")

    monkeypatch.setattr(control, "solve_discrete_are", _fail)
    result = solve_generalized_dare(*args, backend="torch")
    assert torch.isfinite(result).all()


def test_torch_backend_preserves_cpu_device():
    args = _fixture(torch.float64)
    result = solve_generalized_dare(*args, backend="torch")
    assert result.device.type == "cpu"


@pytest.mark.cuda
def test_torch_backend_preserves_cuda_device_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    args = tuple(x.cuda() for x in _fixture(torch.float64))
    result = solve_generalized_dare(*args, backend="torch")
    assert result.device.type == "cuda"
    expected = _scipy_reference(*args)
    torch.testing.assert_close(result, expected, rtol=1e-9, atol=1e-11)


def test_randomized_parity_across_state_and_observation_sizes():
    generator = torch.Generator().manual_seed(1701)
    for n_states in (1, 2, 3, 5, 6):
        for n_observations in range(1, min(n_states, 3) + 1):
            for _ in range(4):
                a = torch.randn(n_states, n_states, generator=generator, dtype=torch.float64)
                radius = torch.linalg.eigvals(a).abs().max().real
                a = a * (0.85 / radius.clamp_min(1e-12))
                c = torch.randn(
                    n_observations, n_states, generator=generator, dtype=torch.float64
                )
                d = n_states + n_observations
                raw = torch.randn(d, d, generator=generator, dtype=torch.float64)
                joint = raw @ raw.T + 0.25 * torch.eye(d, dtype=torch.float64)
                q = joint[:n_states, :n_states]
                s = joint[:n_states, n_states:]
                r = joint[n_states:, n_states:]
                actual = solve_generalized_dare(a, c, q, r, s, backend="torch")
                expected = _scipy_reference(a, c, q, r, s)
                torch.testing.assert_close(actual, expected, rtol=2e-9, atol=2e-11)
                assert _relative_residual(a, c, q, r, s, actual) < 2e-10


def test_invalid_backend_is_rejected():
    with pytest.raises(ValueError, match="backend"):
        solve_generalized_dare(*_fixture(), backend="invalid")


def test_mixed_dtype_is_rejected_by_torch_backend():
    a, c, q, r, s = _fixture(torch.float64)
    with pytest.raises(ValueError, match="same dtype"):
        solve_generalized_dare(a.float(), c, q, r, s, backend="torch")


def test_incompatible_batch_dimensions_are_rejected():
    a, c, q, r, s = _fixture(torch.float64)
    with pytest.raises(ValueError, match="batch"):
        solve_generalized_dare(
            a.repeat(2, 1, 1),
            c.repeat(3, 1, 1),
            q,
            r,
            s,
            backend="torch",
        )


def test_singular_observation_covariance_is_rejected_by_torch_backend():
    a, c, q, r, s = _fixture(torch.float64)
    r = torch.zeros_like(r)
    with pytest.raises(torch.linalg.LinAlgError):
        solve_generalized_dare(a, c, q, r, s, backend="torch")
