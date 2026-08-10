"""Tests for temporal cross-validated state-space order search."""
from types import SimpleNamespace

import numpy as np
import torch

from complextorch import EpochTimeSeriesSplit, StateSpaceOrderSearchCV
from complextorch.selection import TemporalFold


def _data(batch=2, time=100, variables=2):
    generator = torch.Generator().manual_seed(21)
    noise = 0.3 * torch.randn(
        batch, time, variables, generator=generator, dtype=torch.float64
    )
    values = torch.zeros_like(noise)
    for index in range(2, time):
        values[:, index] = (
            0.7 * values[:, index - 1]
            - 0.2 * values[:, index - 2]
            + noise[:, index]
        )
    return values


class _DeterministicSearch(StateSpaceOrderSearchCV):
    """Test double isolating aggregation and selection semantics."""

    def _fit_and_score_fold(self, data, order, fold):
        fold_index = (fold.train_stop - 30) // 10
        losses = {
            1: (4.0, 4.0, 4.0),
            2: (2.0, 1.0, 2.0),
            3: (1.0, 3.0, 1.0),
        }
        return losses[order][fold_index]

    def _bauer_fold(self, training, orders):
        # Deliberately disagree across folds: Bauer is diagnostic only.
        fold_index = (training.shape[1] - 30) // 10
        selected = (1, 3, 1)[fold_index]
        return np.asarray([0.1, 0.2, 0.3]), selected


def test_order_is_selected_from_aggregated_candidate_scores_not_fold_voting():
    cv = EpochTimeSeriesSplit(n_splits=3, test_size=10, min_train_size=30)
    search = _DeterministicSearch(
        orders=(1, 2, 3),
        past_horizon=2,
        cv=cv,
        selection_rule="best",
        refit=False,
    ).fit(torch.zeros(1, 60, 2, dtype=torch.float64))
    assert search.best_order_ == 2
    np.testing.assert_allclose(
        search.mean_test_scores_, [4.0, 5.0 / 3.0, 5.0 / 3.0]
    )
    assert tuple(search.bauer_order_per_fold_) == (1, 3, 1)


def test_real_larimore_search_exposes_expected_shapes_and_refit():
    search = StateSpaceOrderSearchCV(
        orders=(1, 2),
        past_horizon=3,
        cv=EpochTimeSeriesSplit(n_splits=2, test_size=15, min_train_size=60),
        selection_rule="best",
        refit=True,
    ).fit(_data())
    assert search.best_order_ in {1, 2}
    assert search.fold_scores_.shape == (2, 2)
    assert search.bauer_scores_.shape == (2, 2)
    assert search.best_estimator_.n_states_ == search.best_order_
    assert np.isfinite(search.mean_test_scores_).any()


def test_independent_mode_matches_separate_searches_per_trajectory():
    observations = _data(batch=3)
    cv = EpochTimeSeriesSplit(n_splits=2, test_size=15, min_train_size=60)
    batched = StateSpaceOrderSearchCV(
        orders=(1, 2, 3),
        past_horizon=3,
        cv=cv,
        selection_rule="best",
        mode="independent",
        refit=False,
    ).fit(observations)
    separate = [
        StateSpaceOrderSearchCV(
            orders=(1, 2, 3),
            past_horizon=3,
            cv=cv,
            selection_rule="best",
            mode="independent",
            refit=False,
        ).fit(observations[index])
        for index in range(observations.shape[0])
    ]

    assert batched.fold_scores_.shape == (3, 3, 2)
    assert batched.bauer_scores_.shape == (3, 3, 2)
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
        batched.bauer_scores_,
        np.stack([item.bauer_scores_ for item in separate]),
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


def test_independent_mode_refits_each_selected_dimension():
    observations = _data(batch=3)
    search = StateSpaceOrderSearchCV(
        orders=(1, 2, 3),
        past_horizon=3,
        cv=EpochTimeSeriesSplit(n_splits=2, test_size=15, min_train_size=60),
        selection_rule="best",
        mode="independent",
        refit=True,
    ).fit(observations)

    assert isinstance(search.best_estimator_, tuple)
    assert search.best_estimators_ is search.best_estimator_
    assert len(search.best_estimator_) == observations.shape[0]
    for index, estimator in enumerate(search.best_estimator_):
        assert estimator.n_states_ == int(search.best_order_[index])


def test_bauer_diagnostics_allow_order_below_observation_dimension():
    """Bauer diagnostics must allow latent order below output dimension."""
    search = StateSpaceOrderSearchCV(
        orders=(1, 2, 3), past_horizon=2, refit=False
    )
    search._begin_diagnostics(n_orders=3, n_folds=1)
    decomposition = SimpleNamespace(
        values=torch.zeros(1, 20, 3, dtype=torch.float64),
        correlations=torch.tensor(
            [[0.44, 0.09, 0.08, 0.07]], dtype=torch.float64
        ),
        n_columns=6000,
        batch=1,
    )
    search._fold_diagnostics(
        {"decomposition": decomposition}, (1, 2, 3), fold_index=0
    )
    assert np.isfinite(search._bauer_scores[0, 0])
    assert search._bauer_orders[0] == 1


def test_state_space_cv_defaults_to_mle_innovation_covariance():
    """CV and fixed-order Larimore fitting use the same covariance default."""
    assert StateSpaceOrderSearchCV((1,), 2, refit=False).covariance == "mle"


def test_gap_samples_do_not_enter_validation_score_directly():
    search = StateSpaceOrderSearchCV((1,), 1, refit=False)
    system = SimpleNamespace(
        transition=torch.ones(1, 1, dtype=torch.float64),
        observation=torch.ones(1, 1, dtype=torch.float64),
        gain=torch.ones(1, 1, dtype=torch.float64),
    )
    estimator = SimpleNamespace(system_=system)
    observations = torch.tensor(
        [[[1.0], [2.0], [100.0], [3.0]]], dtype=torch.float64
    )
    fold = TemporalFold(train_stop=2, test_start=3, test_stop=4)
    errors = search._innovation_errors(estimator, observations, fold)
    # Gap observations may warm the predictor, but are never themselves scored.
    assert errors.shape == (1, 1, 1)
