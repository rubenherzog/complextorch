import numpy as np
import pytest
import torch

from complextorch import VAROrderSelectionIC, demo_var, simulate_var


def _two_var_trajectories():
    a1, v1 = demo_var(n_variables=3, order=1, dtype=torch.float64)
    a3, v3 = demo_var(n_variables=3, order=3, dtype=torch.float64)
    x1 = simulate_var(a1, v1, n_times=900, burnin=500, seed=101)[0]
    x3 = simulate_var(a3, v3, n_times=900, burnin=500, seed=202)[0]
    return torch.stack((x1, x3))


def test_var_order_selection_independent_batch_matches_separate_fits():
    observations = _two_var_trajectories()
    orders = (1, 2, 3, 4)
    batched = VAROrderSelectionIC(
        orders=orders,
        solver="lstsq",
        mode="independent",
        refit=None,
        device="cpu",
        dtype="float64",
    ).fit(observations)

    separate = [
        VAROrderSelectionIC(
            orders=orders,
            solver="lstsq",
            mode="independent",
            refit=None,
            device="cpu",
            dtype="float64",
        ).fit(observations[index])
        for index in range(observations.shape[0])
    ]

    assert batched.aic_.shape == (2, len(orders))
    assert batched.bic_.shape == (2, len(orders))
    assert batched.hqc_.shape == (2, len(orders))
    assert batched.loglik_.shape == (2, len(orders))
    np.testing.assert_allclose(batched.aic_, np.stack([item.aic_ for item in separate]))
    np.testing.assert_allclose(batched.bic_, np.stack([item.bic_ for item in separate]))
    np.testing.assert_allclose(batched.hqc_, np.stack([item.hqc_ for item in separate]))
    np.testing.assert_allclose(batched.loglik_, np.stack([item.loglik_ for item in separate]))
    np.testing.assert_array_equal(batched.p_aic_, [item.p_aic_ for item in separate])
    np.testing.assert_array_equal(batched.p_bic_, [item.p_bic_ for item in separate])
    np.testing.assert_array_equal(batched.p_hqc_, [item.p_hqc_ for item in separate])


def test_var_order_selection_default_remains_pooled_for_batched_input():
    observations = _two_var_trajectories()
    orders = (1, 2, 3)
    default = VAROrderSelectionIC(
        orders=orders, solver="lstsq", refit=None, device="cpu"
    ).fit(observations)
    explicit = VAROrderSelectionIC(
        orders=orders,
        solver="lstsq",
        mode="pooled",
        refit=None,
        device="cpu",
    ).fit(observations)

    assert default.aic_.shape == (len(orders),)
    assert isinstance(default.p_hqc_, int)
    np.testing.assert_allclose(default.aic_, explicit.aic_)
    np.testing.assert_allclose(default.bic_, explicit.bic_)
    np.testing.assert_allclose(default.hqc_, explicit.hqc_)


def test_var_order_selection_independent_batch_rejects_heterogeneous_refit():
    observations = _two_var_trajectories()
    with pytest.raises(ValueError, match="refit must be None"):
        VAROrderSelectionIC(
            orders=(1, 2, 3),
            solver="lstsq",
            mode="independent",
            refit="hqc",
            device="cpu",
        ).fit(observations)
