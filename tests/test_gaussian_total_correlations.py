"""Scientific and batch-contract tests for Gaussian TC and DTC primitives."""

import math

import pytest
import torch

from complextorch.measures import (
    dual_total_correlation,
    o_information,
    s_information,
    total_correlation,
)


def _spd(batch_shape: tuple[int, ...], n: int, *, dtype: torch.dtype) -> torch.Tensor:
    """Construct deterministic well-conditioned SPD covariance matrices."""
    generator = torch.Generator().manual_seed(1729)
    matrix = torch.randn(*batch_shape, n, n, generator=generator, dtype=dtype)
    eye = torch.eye(n, dtype=dtype).expand(*batch_shape, n, n)
    return matrix @ matrix.transpose(-1, -2) + 0.75 * eye


def _dtc_leave_one_out_reference(covariance: torch.Tensor, base: float) -> torch.Tensor:
    """Evaluate DTC from its independent leave-one-out entropy identity."""
    n = covariance.shape[-1]
    joint_logdet = torch.linalg.slogdet(covariance).logabsdet
    marginal_logdets = []
    for excluded in range(n):
        indices = torch.tensor(
            [index for index in range(n) if index != excluded],
            device=covariance.device,
        )
        marginal = covariance.index_select(-2, indices).index_select(-1, indices)
        marginal_logdets.append(torch.linalg.slogdet(marginal).logabsdet)
    return 0.5 * (
        torch.stack(marginal_logdets, dim=-1).sum(dim=-1)
        - (n - 1) * joint_logdet
    ) / math.log(base)


def test_independent_variables_have_zero_tc_and_dtc() -> None:
    covariance = torch.diag(torch.tensor([0.5, 2.0, 7.0], dtype=torch.float64))

    tc = total_correlation(covariance)
    dtc = dual_total_correlation(covariance)

    torch.testing.assert_close(tc, torch.zeros_like(tc), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(dtc, torch.zeros_like(dtc), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("rho", [-0.8, -0.25, 0.25, 0.8])
def test_bivariate_tc_and_dtc_equal_gaussian_mutual_information(rho: float) -> None:
    covariance = torch.tensor([[1.0, rho], [rho, 1.0]], dtype=torch.float64)
    expected = -0.5 * math.log(1.0 - rho * rho) / math.log(2.0)

    torch.testing.assert_close(
        total_correlation(covariance),
        torch.tensor(expected, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        dual_total_correlation(covariance),
        torch.tensor(expected, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )


def test_dtc_matches_leave_one_out_definition_for_batched_covariances() -> None:
    covariance = _spd((2, 3), 5, dtype=torch.float64)
    expected = _dtc_leave_one_out_reference(covariance, base=math.e)

    actual = dual_total_correlation(covariance, base=math.e)

    assert actual.shape == (2, 3)
    assert actual.dtype == covariance.dtype
    assert actual.device == covariance.device
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)


def test_tc_matches_logdet_definition_for_batched_covariances() -> None:
    covariance = _spd((4,), 6, dtype=torch.float64)
    diagonal = torch.diagonal(covariance, dim1=-2, dim2=-1)
    expected = 0.5 * (
        torch.log(diagonal).sum(dim=-1)
        - torch.linalg.slogdet(covariance).logabsdet
    ) / math.log(10.0)

    actual = total_correlation(covariance, base=10.0)

    assert actual.shape == (4,)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)


def test_o_and_s_information_reuse_tc_dtc_identities() -> None:
    covariance = _spd((3,), 4, dtype=torch.float64)
    tc = total_correlation(covariance)
    dtc = dual_total_correlation(covariance)

    torch.testing.assert_close(o_information(covariance), tc - dtc)
    torch.testing.assert_close(s_information(covariance), tc + dtc)


def test_tc_and_dtc_are_differentiable() -> None:
    raw = torch.tensor(
        [[1.0, 0.2, -0.1], [0.3, 0.9, 0.4], [-0.2, 0.1, 1.1]],
        dtype=torch.float64,
        requires_grad=True,
    )
    covariance = raw @ raw.transpose(-1, -2) + 0.5 * torch.eye(3, dtype=raw.dtype)

    loss = total_correlation(covariance) + dual_total_correlation(covariance)
    loss.backward()

    assert raw.grad is not None
    assert bool(torch.isfinite(raw.grad).all())


@pytest.mark.parametrize("base", [0.0, 1.0, -2.0, math.inf, math.nan])
def test_invalid_log_base_is_rejected(base: float) -> None:
    covariance = torch.eye(3, dtype=torch.float64)
    with pytest.raises(ValueError, match="base"):
        total_correlation(covariance, base=base)


def test_invalid_covariance_shape_and_dtype_are_rejected() -> None:
    with pytest.raises(ValueError, match="shape"):
        total_correlation(torch.ones(3, dtype=torch.float64))
    with pytest.raises(ValueError, match="shape"):
        total_correlation(torch.ones(2, 3, dtype=torch.float64))
    with pytest.raises(TypeError, match="floating-point"):
        total_correlation(torch.eye(3, dtype=torch.int64))
