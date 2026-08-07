"""Shared API, prediction, gap, and Larimore-cache contracts."""
import numpy as np
import torch

import complextorch.selection.selection_state_space as ss_selection
from complextorch import (
    EpochTimeSeriesSplit,
    StateSpaceOrderSearchCV,
    VAROrderSearchCV,
)
from complextorch.selection._temporal import _TemporalOrderSearchCV


def _series(batch=2, time=100, variables=2):
    generator = torch.Generator().manual_seed(31)
    noise = 0.25 * torch.randn(
        batch, time, variables, generator=generator, dtype=torch.float64
    )
    values = torch.zeros_like(noise)
    for index in range(2, time):
        values[:, index] = (
            0.65 * values[:, index - 1]
            - 0.20 * values[:, index - 2]
            + noise[:, index]
        )
    return values


def test_var_and_state_space_search_share_internal_engine_and_api():
    cv = EpochTimeSeriesSplit(n_splits=2, test_size=15, min_train_size=60, gap=3)
    var = VAROrderSearchCV(
        [1, 2], cv=cv, prediction_mode="recursive", gap_mode="embargo",
        refit=False,
    )
    state_space = StateSpaceOrderSearchCV(
        [1, 2], 3, cv=cv, prediction_mode="recursive", gap_mode="embargo",
        refit=False, bauer_diagnostics=False,
    )
    assert isinstance(var, _TemporalOrderSearchCV)
    assert isinstance(state_space, _TemporalOrderSearchCV)
    assert var.get_params()["prediction_mode"] == "recursive"
    assert state_space.get_params()["gap_mode"] == "embargo"


def test_state_space_reuses_one_cva_decomposition_per_fold(monkeypatch):
    calls = 0
    original = ss_selection._larimore_decomposition

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(ss_selection, "_larimore_decomposition", counted)
    cv = EpochTimeSeriesSplit(n_splits=2, test_size=15, min_train_size=60)
    search = StateSpaceOrderSearchCV(
        [1, 2, 3], 3, cv=cv, refit=False, bauer_diagnostics=True
    ).fit(_series())
    assert calls == 2
    assert search.fold_scores_.shape == (3, 2)
    assert search.bauer_scores_.shape == (3, 2)


def test_var_gap_modes_have_distinct_semantics():
    data = _series(batch=1)
    data[:, 62:65] += 20.0
    cv = EpochTimeSeriesSplit(n_splits=1, test_size=15, min_train_size=62, gap=3)
    warmup = VAROrderSearchCV(
        [1], cv=cv, prediction_mode="rolling", gap_mode="warmup", refit=False
    ).fit(data)
    embargo = VAROrderSearchCV(
        [1], cv=cv, prediction_mode="rolling", gap_mode="embargo", refit=False
    ).fit(data)
    assert not np.isclose(warmup.fold_scores_[0, 0], embargo.fold_scores_[0, 0])


def test_both_searches_select_one_global_order_across_folds():
    data = _series()
    cv = EpochTimeSeriesSplit(n_splits=2, test_size=15, min_train_size=60)
    var = VAROrderSearchCV([1, 2, 3], cv=cv, refit=True).fit(data)
    state_space = StateSpaceOrderSearchCV(
        [1, 2, 3], 3, cv=cv, refit=True
    ).fit(data)
    assert int(var.best_order_) in {1, 2, 3}
    assert int(state_space.best_order_) in {1, 2, 3}
    assert var.best_estimator_.order == var.best_order_
    assert state_space.best_estimator_.n_states == state_space.best_order_
