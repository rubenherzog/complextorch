import torch

from complextorch import (
    InnovationsStateSpace,
    build_var_system,
    downsample_innovations_state_space,
    var_to_innovations_state_space,
    varma_to_innovations_state_space,
)


def test_varma_q0_matches_existing_var_innovations_form():
    dtype = torch.float64
    coefficients = torch.tensor(
        [
            [[0.45, 0.10], [-0.05, 0.30]],
            [[0.10, 0.00], [0.04, -0.08]],
        ],
        dtype=dtype,
    )
    covariance = torch.tensor([[1.2, 0.15], [0.15, 0.8]], dtype=dtype)

    system = build_var_system(coefficients, covariance)
    reference = var_to_innovations_state_space(system)
    converted = varma_to_innovations_state_space(
        coefficients, None, covariance
    )

    torch.testing.assert_close(converted.transition, reference.transition[0])
    torch.testing.assert_close(converted.observation, reference.observation[0])
    torch.testing.assert_close(converted.gain, reference.gain[0])
    torch.testing.assert_close(
        converted.innovation_covariance, reference.innovation_covariance[0]
    )


def test_varma_aoki_realization_matches_reference_blocks_without_inverse():
    dtype = torch.float64
    ar = torch.tensor([[[0.30, 0.00], [0.00, 0.20]]], dtype=dtype)
    ma = torch.tensor([[[0.10, 0.00], [0.00, -0.05]]], dtype=dtype)
    covariance = torch.diag(torch.tensor([1.0, 4.0], dtype=dtype))
    b0 = torch.diag(torch.tensor([2.0, 0.5], dtype=dtype))

    converted = varma_to_innovations_state_space(
        ar, ma, covariance, zero_lag_ma=b0
    )

    expected_transition = torch.tensor(
        [
            [0.30, 0.00, 0.10, 0.00],
            [0.00, 0.20, 0.00, -0.05],
            [0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00],
        ],
        dtype=dtype,
    )
    expected_observation = torch.tensor(
        [
            [0.30, 0.00, 0.10, 0.00],
            [0.00, 0.20, 0.00, -0.05],
        ],
        dtype=dtype,
    )
    expected_gain = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.0],
            [0.0, 2.0],
        ],
        dtype=dtype,
    )
    expected_innovation = torch.diag(torch.tensor([4.0, 1.0], dtype=dtype))

    torch.testing.assert_close(converted.transition, expected_transition)
    torch.testing.assert_close(converted.observation, expected_observation)
    torch.testing.assert_close(converted.gain, expected_gain)
    torch.testing.assert_close(converted.innovation_covariance, expected_innovation)


def test_varma_supports_compatible_batch_broadcasting_and_dtype():
    dtype = torch.float32
    ar = torch.tensor(
        [
            [[[0.4]]],
            [[[0.7]]],
        ],
        dtype=dtype,
    ).reshape(2, 1, 1, 1)
    ma = torch.tensor([[[0.2]]], dtype=dtype)
    covariance = torch.tensor([[1.5]], dtype=dtype)

    converted = varma_to_innovations_state_space(ar, ma, covariance)

    assert converted.transition.shape == (2, 2, 2)
    assert converted.observation.shape == (2, 1, 2)
    assert converted.gain.shape == (2, 2, 1)
    assert converted.innovation_covariance.shape == (2, 1, 1)
    assert converted.transition.dtype == dtype
    assert converted.transition.device == ar.device


def test_varma_rejects_singular_zero_lag_ma_when_ma_state_is_present():
    ar = torch.tensor([[[0.4]]], dtype=torch.float64)
    ma = torch.tensor([[[0.2]]], dtype=torch.float64)
    covariance = torch.tensor([[1.0]], dtype=torch.float64)
    b0 = torch.tensor([[0.0]], dtype=torch.float64)

    try:
        varma_to_innovations_state_space(
            ar, ma, covariance, zero_lag_ma=b0
        )
    except ValueError as exc:
        assert "nonsingular" in str(exc)
    else:
        raise AssertionError("singular B0 should be rejected")


def test_downsample_factor_one_is_exact_identity():
    system = InnovationsStateSpace(
        transition=torch.tensor([[0.5]], dtype=torch.float64),
        observation=torch.tensor([[0.5]], dtype=torch.float64),
        gain=torch.tensor([[1.0]], dtype=torch.float64),
        innovation_covariance=torch.tensor([[2.0]], dtype=torch.float64),
    )

    assert downsample_innovations_state_space(system, 1) is system


def test_downsample_scalar_ar1_has_exact_transition_and_innovation_variance():
    dtype = torch.float64
    a = 0.5
    variance = 2.0
    factor = 3
    system = InnovationsStateSpace(
        transition=torch.tensor([[a]], dtype=dtype),
        observation=torch.tensor([[a]], dtype=dtype),
        gain=torch.tensor([[1.0]], dtype=dtype),
        innovation_covariance=torch.tensor([[variance]], dtype=dtype),
    )

    downsampled = downsample_innovations_state_space(system, factor)

    expected_variance = variance * sum(a ** (2 * lag) for lag in range(factor))
    torch.testing.assert_close(
        downsampled.transition,
        torch.tensor([[a**factor]], dtype=dtype),
        rtol=1e-10,
        atol=1e-12,
    )
    torch.testing.assert_close(
        downsampled.innovation_covariance,
        torch.tensor([[expected_variance]], dtype=dtype),
        rtol=1e-9,
        atol=1e-11,
    )
    torch.testing.assert_close(downsampled.observation, system.observation)


def test_downsample_batch_preserves_batch_dtype_and_device():
    dtype = torch.float64
    system = InnovationsStateSpace(
        transition=torch.tensor([[[0.35]], [[0.60]]], dtype=dtype),
        observation=torch.tensor([[[0.35]], [[0.60]]], dtype=dtype),
        gain=torch.ones((2, 1, 1), dtype=dtype),
        innovation_covariance=torch.tensor([[[1.0]], [[1.5]]], dtype=dtype),
    )

    downsampled = downsample_innovations_state_space(system, 2)

    assert downsampled.transition.shape == (2, 1, 1)
    assert downsampled.observation.shape == (2, 1, 1)
    assert downsampled.gain.shape == (2, 1, 1)
    assert downsampled.innovation_covariance.shape == (2, 1, 1)
    assert downsampled.transition.dtype == dtype
    assert downsampled.transition.device == system.transition.device
    torch.testing.assert_close(
        downsampled.transition.squeeze(-1).squeeze(-1),
        torch.tensor([0.35**2, 0.60**2], dtype=dtype),
    )


def test_downsample_rejects_invalid_factor():
    system = InnovationsStateSpace(
        transition=torch.tensor([[0.5]], dtype=torch.float64),
        observation=torch.tensor([[0.5]], dtype=torch.float64),
        gain=torch.tensor([[1.0]], dtype=torch.float64),
        innovation_covariance=torch.tensor([[1.0]], dtype=torch.float64),
    )

    for factor in (0, -1, 1.5, True):
        try:
            downsample_innovations_state_space(system, factor)  # type: ignore[arg-type]
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid factor {factor!r} should be rejected")
