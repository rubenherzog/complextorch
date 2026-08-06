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
    np.testing.assert_allclose(search.mean_test_scores_, [4.0, 5.0 / 3.0, 5.0 / 3.0])
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


def test_independent_mode_selects_one_global_order():
    search = StateSpaceOrderSearchCV(
        orders=(1, 2),
        past_horizon=3,
        cv=EpochTimeSeriesSplit(n_splits=2, test_size=15, min_train_size=60),
        mode="independent",
        refit=False,
    ).fit(_data(batch=3))
    assert isinstance(search.best_order_, int)
    assert search.fold_scores_.shape == (2, 2)


def test_gap_samples_do_not_enter_validation_score_directly():
    search = StateSpaceOrderSearchCV((1,), 1, refit=False)
    system = SimpleNamespace(
        transition=torch.ones(1, 1, dtype=torch.float64),
        observation=torch.ones(1, 1, dtype=torch.float64),
        gain=torch.ones(1, 1, dtype=torch.float64),
    )
    estimator = SimpleNamespace(system_=system)
    observations = torch.tensor([[[1.0], [2.0], [100.0], [3.0]]], dtype=torch.float64)
    fold = TemporalFold(train_stop=2, test_start=3, test_stop=4)
    errors = search._innovation_errors(estimator, observations, fold)
    # The current implementation warms the rolling predictor chronologically;
    # only the final held-out innovation is returned.
    assert errors.shape == (1, 1, 1)
