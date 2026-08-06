"""Tests for Bauer SVC and Larimore canonical-variate state-order estimation."""

import numpy as np
import pytest
import torch

from complextorch import (
    LarimoreStateSpaceOrder,
    bauer_svc,
    larimore_state_space_order,
)


def _numpy_larimore(values: np.ndarray, horizon: int, ridge: float = 1e-12):
    """Small NumPy reference matching ComplexBox's pooled trial convention."""

    if values.ndim == 2:
        values = values[None]
    values = values - values.mean(axis=1, keepdims=True)
    batch, n_times, n_variables = values.shape
    n_columns = n_times - 2 * horizon + 1
    past = np.zeros((horizon * n_variables, batch * n_columns))
    future = np.zeros_like(past)
    column = 0
    for trial in range(batch):
        for time in range(horizon, n_times - horizon + 1):
            for lag in range(horizon):
                past[
                    lag * n_variables : (lag + 1) * n_variables,
                    column,
                ] = values[trial, time - 1 - lag]
                future[
                    lag * n_variables : (lag + 1) * n_variables,
                    column,
                ] = values[trial, time + lag]
            column += 1
    n_effective = past.shape[1]
    covariance_past = past @ past.T / n_effective
    covariance_future = future @ future.T / n_effective
    cross_covariance = past @ future.T / n_effective
    cholesky_past = np.linalg.cholesky(
        covariance_past + ridge * np.eye(covariance_past.shape[0])
    )
    cholesky_future = np.linalg.cholesky(
        covariance_future + ridge * np.eye(covariance_future.shape[0])
    )
    whitened = np.linalg.solve(
        cholesky_future, cross_covariance.T
    ) @ np.linalg.inv(cholesky_past.T)
    return np.linalg.svd(whitened, compute_uv=False), n_effective


def test_bauer_svc_matches_closed_form():
    """The Torch criterion must equal the ComplexBox/Bauer expression."""

    correlations = torch.tensor([0.95, 0.70, 0.10, 0.02], dtype=torch.float64)
    best, criterion = bauer_svc(
        correlations,
        n_observations=1,
        n_effective=200,
    )
    orders = torch.arange(1, 5, dtype=torch.float64)
    omitted = torch.tensor([0.70, 0.10, 0.02, 0.0], dtype=torch.float64)
    expected = omitted.square() + 2.0 * orders * np.log(200.0) / 200.0
    torch.testing.assert_close(criterion, expected)
    assert int(best) == int(torch.argmin(expected)) + 1


def test_bauer_svc_vectorizes_over_batches():
    """Leading dimensions are independent model-order problems."""

    correlations = torch.tensor(
        [[0.9, 0.5, 0.08], [0.95, 0.2, 0.01]],
        dtype=torch.float32,
    )
    best, criterion = bauer_svc(
        correlations,
        n_observations=1,
        n_effective=torch.tensor([100.0, 400.0]),
    )
    assert best.shape == (2,)
    assert criterion.shape == (2, 3)
    assert criterion.dtype == torch.float32
    for batch in range(2):
        scalar_best, scalar_curve = bauer_svc(
            correlations[batch],
            n_observations=1,
            n_effective=[100, 400][batch],
        )
        torch.testing.assert_close(criterion[batch], scalar_curve)
        assert int(best[batch]) == int(scalar_best)


def test_larimore_pooled_matches_numpy_complexbox_convention():
    """Hankel orientation, whitening and singular values match NumPy parity."""

    generator = torch.Generator().manual_seed(20260806)
    observations = torch.randn(
        3, 90, 2, generator=generator, dtype=torch.float64
    )
    result = larimore_state_space_order(
        observations,
        past_horizon=4,
        mode="pooled",
    )
    expected_correlations, expected_n = _numpy_larimore(
        observations.numpy(), horizon=4
    )
    np.testing.assert_allclose(
        result.canonical_correlations.numpy(),
        expected_correlations,
        rtol=1e-9,
        atol=1e-10,
    )
    assert int(result.n_effective) == expected_n
    expected_best, expected_curve = bauer_svc(
        torch.from_numpy(expected_correlations),
        n_observations=2,
        n_effective=expected_n,
    )
    assert int(result.best_order) == int(expected_best)
    torch.testing.assert_close(result.criterion, expected_curve)


def test_larimore_independent_matches_single_trajectory_calls():
    """Independent mode must be equivalent to one call per batch item."""

    generator = torch.Generator().manual_seed(17)
    observations = torch.randn(
        4, 75, 2, generator=generator, dtype=torch.float64
    )
    batched = larimore_state_space_order(
        observations,
        past_horizon=3,
        mode="independent",
    )
    assert batched.best_order.shape == (4,)
    assert batched.criterion.shape[0] == 4
    assert batched.canonical_correlations.shape == (4, 6)
    for batch in range(observations.shape[0]):
        single = larimore_state_space_order(
            observations[batch],
            past_horizon=3,
            mode="pooled",
        )
        assert int(batched.best_order[batch]) == int(single.best_order)
        torch.testing.assert_close(
            batched.criterion[batch], single.criterion
        )
        torch.testing.assert_close(
            batched.canonical_correlations[batch],
            single.canonical_correlations,
        )


def test_larimore_estimator_exposes_fitted_attributes():
    """The estimator follows scikit-learn's trailing-underscore convention."""

    observations = torch.randn(100, 3, dtype=torch.float64)
    estimator = LarimoreStateSpaceOrder(
        4,
        future_horizon=5,
        mode="pooled",
    ).fit(observations)
    assert estimator.result_.best_order is estimator.best_order_
    assert estimator.candidate_orders_.ndim == 1
    assert estimator.criterion_.shape[-1] == estimator.candidate_orders_.numel()
    assert estimator.canonical_correlations_.shape[-1] == 12
    assert torch.isclose(
        estimator.normalized_canonical_correlations_[0],
        torch.tensor(1.0, dtype=torch.float64),
    )


def test_larimore_preserves_requested_dtype_and_validates_inputs():
    """Torch execution contracts include dtype and explicit input validation."""

    observations = torch.randn(60, 2)
    result = larimore_state_space_order(
        observations,
        past_horizon=3,
        dtype="float32",
    )
    assert result.criterion.dtype == torch.float32
    assert result.canonical_correlations.dtype == torch.float32

    with pytest.raises(ValueError):
        larimore_state_space_order(observations, past_horizon=0)
    with pytest.raises(ValueError):
        larimore_state_space_order(observations, past_horizon=30)
    with pytest.raises(ValueError):
        larimore_state_space_order(observations, past_horizon=3, mode="bad")
    with pytest.raises(ValueError):
        bauer_svc(torch.tensor([0.8, 0.2]), 2, 100, min_order=1)
    with pytest.raises(ValueError):
        bauer_svc(torch.tensor([0.8, 0.2]), 3, 100)
