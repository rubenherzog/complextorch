import numpy as np
import torch

from complextorch import (
    EpochTimeSeriesSplit,
    VAROrderSearchCV,
    VAROrderSelectionIC,
    demo_var,
    simulate_var,
)


def test_cv_information_criteria_match_selection_ic_on_each_training_fold():
    coefficients, covariance = demo_var(
        n_variables=3,
        order=2,
        dtype=torch.float64,
    )
    observations = simulate_var(
        coefficients,
        covariance,
        n_times=720,
        burnin=500,
        seed=31,
    )[0]
    orders = (1, 2, 3)
    splitter = EpochTimeSeriesSplit(
        n_splits=3,
        test_size=100,
        min_train_size=400,
    )
    search = VAROrderSearchCV(
        orders=orders,
        cv=splitter,
        scoring="nll",
        selection_rule="best",
        solver="lwr",
        mode="independent",
        device="cpu",
        dtype="float64",
        refit=False,
    ).fit(observations)

    folds = tuple(splitter.split(observations.shape[0], min_order=max(orders)))
    assert search.train_aic_.shape == (len(orders), len(folds))
    assert search.train_bic_.shape == (len(orders), len(folds))
    assert search.train_hqc_.shape == (len(orders), len(folds))

    for fold_index, fold in enumerate(folds):
        reference = VAROrderSelectionIC(
            orders=orders,
            solver="lwr",
            device="cpu",
            dtype="float64",
            refit=None,
        ).fit(observations[: fold.train_stop])
        np.testing.assert_allclose(
            search.train_aic_[:, fold_index], reference.aic_, rtol=1e-10, atol=1e-12
        )
        np.testing.assert_allclose(
            search.train_bic_[:, fold_index], reference.bic_, rtol=1e-10, atol=1e-12
        )
        np.testing.assert_allclose(
            search.train_hqc_[:, fold_index], reference.hqc_, rtol=1e-10, atol=1e-12
        )


def test_cv_information_criteria_are_diagnostics_not_selection_scores():
    coefficients, covariance = demo_var(
        n_variables=3,
        order=2,
        dtype=torch.float64,
    )
    observations = simulate_var(
        coefficients,
        covariance,
        n_times=600,
        burnin=400,
        seed=41,
    )[0]
    search = VAROrderSearchCV(
        orders=(1, 2, 3),
        cv=EpochTimeSeriesSplit(n_splits=2, test_size=100, min_train_size=350),
        scoring="rmse",
        selection_rule="best",
        solver="lstsq",
        device="cpu",
        dtype="float64",
        refit=False,
    ).fit(observations)

    expected_best = min(search.result_.scores, key=lambda result: result.mean_score).order
    assert search.best_order_ == expected_best
    assert np.allclose(
        search.mean_train_aic_,
        np.nanmean(search.train_aic_, axis=1),
    )
    assert np.allclose(
        search.mean_train_bic_,
        np.nanmean(search.train_bic_, axis=1),
    )
    assert np.allclose(
        search.mean_train_hqc_,
        np.nanmean(search.train_hqc_, axis=1),
    )
    for record in search.cv_results_:
        assert "mean_train_aic" in record
        assert "fold_train_bic" in record
        assert "fold_train_hqc" in record
