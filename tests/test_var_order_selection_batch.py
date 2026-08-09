import numpy as np
import pytest
import torch

from complextorch import (
    EpochTimeSeriesSplit,
    VAR,
    VAROrderSearchCV,
    VAROrderSelectionIC,
    demo_var,
    simulate_var,
)


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


@pytest.mark.parametrize("scoring", ["nll", "rmse"])
def test_var_cv_independent_batch_matches_separate_searches(scoring):
    observations = _heterogeneous_var_batch()
    orders = (1, 2, 3, 4)
    cv = EpochTimeSeriesSplit(n_splits=3, test_size=60, min_train_size=280)
    common = dict(
        orders=orders,
        cv=cv,
        scoring=scoring,
        selection_rule="best",
        solver="lwr",
        mode="independent",
        refit=False,
        device="cpu",
        dtype="float64",
    )
    batched = VAROrderSearchCV(**common).fit(observations)
    separate = [
        VAROrderSearchCV(**common).fit(observations[index])
        for index in range(observations.shape[0])
    ]

    expected_score_shape = (observations.shape[0], len(orders), cv.n_splits)
    expected_curve_shape = (observations.shape[0], len(orders))
    assert batched.fold_scores_.shape == expected_score_shape
    assert batched.mean_test_scores_.shape == expected_curve_shape
    assert batched.standard_error_.shape == expected_curve_shape
    assert batched.failed_folds_.shape == expected_curve_shape
    assert batched.train_aic_.shape == expected_score_shape
    assert batched.train_bic_.shape == expected_score_shape
    assert batched.train_hqc_.shape == expected_score_shape
    assert batched.best_order_.shape == (observations.shape[0],)

    np.testing.assert_allclose(
        batched.fold_scores_,
        np.stack([item.fold_scores_ for item in separate]),
        rtol=2e-7,
        atol=2e-9,
    )
    np.testing.assert_allclose(
        batched.mean_test_scores_,
        np.stack([item.mean_test_scores_ for item in separate]),
        rtol=2e-7,
        atol=2e-9,
    )
    np.testing.assert_allclose(
        batched.train_aic_,
        np.stack([item.train_aic_ for item in separate]),
        rtol=2e-7,
        atol=2e-9,
    )
    np.testing.assert_allclose(
        batched.train_bic_,
        np.stack([item.train_bic_ for item in separate]),
        rtol=2e-7,
        atol=2e-9,
    )
    np.testing.assert_allclose(
        batched.train_hqc_,
        np.stack([item.train_hqc_ for item in separate]),
        rtol=2e-7,
        atol=2e-9,
    )
    np.testing.assert_array_equal(
        batched.best_order_, [item.best_order_ for item in separate]
    )

    for order_index, record in enumerate(batched.result_.scores):
        np.testing.assert_allclose(
            record.mean_score, batched.mean_test_scores_[:, order_index]
        )
        np.testing.assert_allclose(
            record.fold_scores, batched.fold_scores_[:, order_index, :]
        )


def test_var_cv_independent_batch_refits_each_selected_order():
    observations = _heterogeneous_var_batch()
    search = VAROrderSearchCV(
        orders=(1, 2, 3, 4),
        cv=EpochTimeSeriesSplit(
            n_splits=2, test_size=60, min_train_size=300
        ),
        selection_rule="best",
        solver="lwr",
        mode="independent",
        refit=True,
        device="cpu",
        dtype="float64",
    ).fit(observations)

    assert isinstance(search.best_estimator_, tuple)
    assert search.best_estimators_ is search.best_estimator_
    assert len(search.best_estimator_) == observations.shape[0]
    for index, estimator in enumerate(search.best_estimator_):
        assert estimator.order == int(search.best_order_[index])
        assert estimator.coef_.shape[0] == 1
