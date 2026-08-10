"""N4SID and estimator-generic state-space temporal CV tests."""
import numpy as np
import torch

from complextorch import EpochTimeSeriesSplit, N4SID, StateSpaceOrderSearchCV


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


def _latent_ssm_data(batch=3, time=600, seed=17):
    generator = torch.Generator().manual_seed(seed)
    transition = torch.tensor(
        [[0.82, 0.12], [-0.08, 0.62]], dtype=torch.float64
    )
    observation = torch.tensor(
        [[1.0, 0.2], [0.1, 1.0], [0.7, -0.3], [-0.4, 0.6]],
        dtype=torch.float64,
    )
    process_factor = torch.linalg.cholesky(
        0.08 * torch.eye(2, dtype=torch.float64)
    )
    observation_factor = torch.linalg.cholesky(
        0.01 * torch.eye(4, dtype=torch.float64)
    )
    trajectories = []
    for _ in range(batch):
        state = torch.zeros(2, dtype=torch.float64)
        values = []
        for index in range(time + 200):
            state = transition @ state + process_factor @ torch.randn(
                2, generator=generator, dtype=torch.float64
            )
            value = observation @ state + observation_factor @ torch.randn(
                4, generator=generator, dtype=torch.float64
            )
            if index >= 200:
                values.append(value)
        trajectories.append(torch.stack(values))
    return torch.stack(trajectories), observation


def test_n4sid_cv_recovers_true_order_against_underfit_candidate():
    observations, _ = _latent_ssm_data(batch=3, time=600)
    search = StateSpaceOrderSearchCV(
        orders=(1, 2),
        past_horizon=5,
        method="n4sid",
        mode="pooled",
        cv=EpochTimeSeriesSplit(
            n_splits=3, test_size=80, min_train_size=300
        ),
        selection_rule="best",
        refit=True,
        bauer_diagnostics=False,
    ).fit(observations)

    assert search.best_order_ == 2
    assert search.best_estimator_.n_states_ == 2
    assert search.method_ == "n4sid"
    assert search.mean_test_scores_[1] < search.mean_test_scores_[0]


def test_n4sid_recovers_observation_subspace_at_selected_dimension():
    observations, true_observation = _latent_ssm_data(batch=3, time=700)
    search = StateSpaceOrderSearchCV(
        orders=(1, 2),
        past_horizon=5,
        method="n4sid",
        mode="pooled",
        cv=EpochTimeSeriesSplit(
            n_splits=3, test_size=80, min_train_size=320
        ),
        selection_rule="best",
        refit=True,
        bauer_diagnostics=False,
    ).fit(observations)

    estimated = search.best_estimator_.observation_
    if estimated.ndim == 3:
        estimated = estimated[0]
    true_basis = torch.linalg.qr(true_observation).Q[:, :2]
    estimated_basis = torch.linalg.qr(estimated).Q[:, :2]
    singular_values = torch.linalg.svdvals(
        true_basis.transpose(-1, -2) @ estimated_basis
    )
    principal_angles = torch.acos(singular_values.clamp(-1.0, 1.0))
    assert float(principal_angles.max()) < 0.05


def test_n4sid_independent_batch_matches_separate_searches():
    observations = _data(batch=3, time=120)
    cv = EpochTimeSeriesSplit(n_splits=2, test_size=15, min_train_size=70)
    common = dict(
        orders=(1, 2, 3),
        past_horizon=3,
        method="n4sid",
        mode="independent",
        cv=cv,
        selection_rule="best",
        refit=False,
        bauer_diagnostics=False,
    )
    batched = StateSpaceOrderSearchCV(**common).fit(observations)
    separate = [
        StateSpaceOrderSearchCV(**common).fit(observations[index])
        for index in range(observations.shape[0])
    ]

    assert batched.fold_scores_.shape == (3, 3, 2)
    assert batched.mean_test_scores_.shape == (3, 3)
    np.testing.assert_allclose(
        batched.fold_scores_,
        np.stack([item.fold_scores_ for item in separate]),
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_array_equal(
        batched.best_order_, [item.best_order_ for item in separate]
    )


def test_estimator_prototype_uses_same_temporal_search_infrastructure():
    observations = _data(batch=2, time=120)
    cv = EpochTimeSeriesSplit(n_splits=2, test_size=15, min_train_size=70)
    prototype = N4SID(
        n_states=1,
        block_rows=3,
        ridge=1e-8,
        mode="pooled",
        dtype="float64",
    )
    via_estimator = StateSpaceOrderSearchCV(
        orders=(1, 2),
        past_horizon=3,
        estimator=prototype,
        mode="pooled",
        cv=cv,
        selection_rule="best",
        refit=True,
        bauer_diagnostics=False,
    ).fit(observations)
    via_method = StateSpaceOrderSearchCV(
        orders=(1, 2),
        past_horizon=3,
        method="n4sid",
        mode="pooled",
        cv=cv,
        selection_rule="best",
        refit=True,
        bauer_diagnostics=False,
    ).fit(observations)

    np.testing.assert_allclose(
        via_estimator.fold_scores_,
        via_method.fold_scores_,
        rtol=2e-6,
        atol=2e-7,
    )
    assert via_estimator.best_order_ == via_method.best_order_
    assert via_estimator.best_estimator_.n_states_ == via_estimator.best_order_
    assert prototype.n_states == 1


def test_n4sid_independent_batch_refits_selected_dimensions():
    observations = _data(batch=3, time=120)
    search = StateSpaceOrderSearchCV(
        orders=(1, 2, 3),
        past_horizon=3,
        method="n4sid",
        mode="independent",
        cv=EpochTimeSeriesSplit(n_splits=2, test_size=15, min_train_size=70),
        selection_rule="best",
        refit=True,
        bauer_diagnostics=False,
    ).fit(observations)

    assert isinstance(search.best_estimator_, tuple)
    assert search.best_estimators_ is search.best_estimator_
    assert len(search.best_estimator_) == observations.shape[0]
    for index, estimator in enumerate(search.best_estimator_):
        assert estimator.n_states_ == int(search.best_order_[index])
        assert estimator.states_.shape[0] == 1
