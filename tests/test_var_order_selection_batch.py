import numpy as np
import pytest
import torch

from complextorch import VAR, VAROrderSelectionIC, demo_var, simulate_var


SOLVERS = ("auto", "lstsq", "pinv", "cholesky", "lwr")


def _heterogeneous_var_batch():
    trajectories = []
    for index, order in enumerate((1, 2, 3)):
        coefficients, covariance = demo_var(
            n_variables=3, order=order, dtype=torch.float64
        )
        sample = simulate_var(
            coefficients,
            covariance,
            n_times=500,
            burnin=300,
            seed=101 + index,
        )[0]
        trajectories.append(sample)
    return torch.stack(trajectories)


@pytest.mark.parametrize("solver", SOLVERS)
def test_var_independent_batch_matches_separate_fits_for_every_solver(solver):
    generator = torch.Generator().manual_seed(44)
    observations = torch.randn(
        (5, 180, 4), generator=generator, dtype=torch.float64
    )
    batched = VAR(
        order=3,
        mode="independent",
        solver=solver,
        stability="ignore",
        device="cpu",
        dtype="float64",
    ).fit(observations)
    separate = [
        VAR(
            order=3,
            mode="independent",
            solver=solver,
            stability="ignore",
            device="cpu",
            dtype="float64",
        ).fit(observations[index])
        for index in range(observations.shape[0])
    ]

    torch.testing.assert_close(
        batched.coef_, torch.cat([item.coef_ for item in separate]), rtol=2e-7, atol=2e-9
    )
    torch.testing.assert_close(
        batched.intercept_,
        torch.cat([item.intercept_ for item in separate]),
        rtol=2e-7,
        atol=2e-9,
    )
    torch.testing.assert_close(
        batched.noise_covariance_,
        torch.cat([item.noise_covariance_ for item in separate]),
        rtol=2e-7,
        atol=2e-9,
    )
    torch.testing.assert_close(
        batched.residuals_,
        torch.cat([item.residuals_ for item in separate]),
        rtol=2e-7,
        atol=2e-9,
    )


@pytest.mark.parametrize("solver", SOLVERS)
def test_var_order_selection_independent_batch_matches_separate_fits(solver):
    observations = _heterogeneous_var_batch()
    orders = (1, 2, 3, 4)
    batched = VAROrderSelectionIC(
        orders=orders,
        solver=solver,
        mode="independent",
        refit=None,
        device="cpu",
        dtype="float64",
    ).fit(observations)

    separate = [
        VAROrderSelectionIC(
            orders=orders,
            solver=solver,
            mode="independent",
            refit=None,
            device="cpu",
            dtype="float64",
        ).fit(observations[index])
        for index in range(observations.shape[0])
    ]

    expected_shape = (observations.shape[0], len(orders))
    assert batched.aic_.shape == expected_shape
    assert batched.bic_.shape == expected_shape
    assert batched.hqc_.shape == expected_shape
    assert batched.loglik_.shape == expected_shape
    np.testing.assert_allclose(
        batched.aic_, np.stack([item.aic_ for item in separate]), rtol=2e-7, atol=2e-9
    )
    np.testing.assert_allclose(
        batched.bic_, np.stack([item.bic_ for item in separate]), rtol=2e-7, atol=2e-9
    )
    np.testing.assert_allclose(
        batched.hqc_, np.stack([item.hqc_ for item in separate]), rtol=2e-7, atol=2e-9
    )
    np.testing.assert_allclose(
        batched.loglik_,
        np.stack([item.loglik_ for item in separate]),
        rtol=2e-7,
        atol=2e-9,
    )
    np.testing.assert_array_equal(batched.p_aic_, [item.p_aic_ for item in separate])
    np.testing.assert_array_equal(batched.p_bic_, [item.p_bic_ for item in separate])
    np.testing.assert_array_equal(batched.p_hqc_, [item.p_hqc_ for item in separate])

    for estimator in batched.candidate_estimators_.values():
        assert estimator.coef_.shape[0] == observations.shape[0]


def test_var_order_selection_default_remains_pooled_for_batched_input():
    observations = _heterogeneous_var_batch()
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
    observations = _heterogeneous_var_batch()
    with pytest.raises(ValueError, match="refit must be None"):
        VAROrderSelectionIC(
            orders=(1, 2, 3),
            solver="lstsq",
            mode="independent",
            refit="hqc",
            device="cpu",
        ).fit(observations)


def test_var_order_selection_independent_single_series_can_refit():
    observations = _heterogeneous_var_batch()[0]
    selector = VAROrderSelectionIC(
        orders=(1, 2, 3),
        solver="lwr",
        mode="independent",
        refit="hqc",
        device="cpu",
    ).fit(observations)
    assert isinstance(selector.best_order_, int)
    assert selector.best_estimator_.coef_.shape[0] == 1
