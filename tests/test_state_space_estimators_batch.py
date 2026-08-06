"""Batch contracts for state-space estimators."""
import torch

from complextorch import (
    LarimoreStateSpace,
    LinearGaussianEM,
    N4SID,
    StateSpaceOrderSelection,
)
from complextorch.representations import StateSpaceModel


def _data(batch=3, time=120, variables=2):
    generator = torch.Generator().manual_seed(17)
    noise = torch.randn(batch, time, variables, generator=generator, dtype=torch.float64)
    values = torch.zeros_like(noise)
    for t in range(1, time):
        values[:, t] = 0.65 * values[:, t - 1] + noise[:, t]
    return values


def test_n4sid_pooled_and_independent_batch_contracts():
    x = _data()
    pooled = N4SID(2, block_rows=5, mode="pooled").fit(x)
    independent = N4SID(2, block_rows=5, mode="independent").fit(x)
    assert pooled.transition_.shape == (2, 2)
    assert independent.transition_.shape == (3, 2, 2)
    assert pooled.states_.shape[0] == 3
    assert independent.states_.shape[0] == 3


def test_larimore_pooled_and_independent_batch_contracts():
    x = _data()
    pooled = LarimoreStateSpace(2, 5, mode="pooled").fit(x)
    independent = LarimoreStateSpace(2, 5, mode="independent").fit(x)
    assert pooled.transition_.shape == (2, 2)
    assert pooled.kalman_gain_.shape == (2, 2)
    assert independent.transition_.shape == (3, 2, 2)
    assert independent.kalman_gain_.shape == (3, 2, 2)
    assert pooled.states_.shape[0] == 3


def test_state_space_selector_refits_larimore_in_pooled_mode():
    selector = StateSpaceOrderSelection(5, refit=True).fit(_data())
    assert isinstance(selector.best_estimator_, LarimoreStateSpace)
    assert selector.best_estimator_.n_states_ == int(selector.best_order_)


def test_linear_gaussian_em_accepts_batched_trajectories():
    x = _data(batch=2, time=60)
    system = StateSpaceModel(
        transition=torch.eye(2, dtype=torch.float64) * 0.5,
        observation=torch.eye(2, dtype=torch.float64),
        process_covariance=torch.eye(2, dtype=torch.float64),
        observation_covariance=torch.eye(2, dtype=torch.float64),
        state_covariance=torch.eye(2, dtype=torch.float64),
    )
    pooled = LinearGaussianEM(system, n_iter=2, mode="pooled").fit(x)
    independent = LinearGaussianEM(system, n_iter=2, mode="independent").fit(x)
    assert pooled.system_.transition.shape == (2, 2)
    assert independent.system_.transition.shape == (2, 2, 2)
    assert pooled.trajectory_log_likelihood_history_.shape == (2, 2)
    assert independent.log_likelihood_history_.shape == (2, 2)


def test_pooled_estimators_do_not_create_between_trial_transitions():
    x = _data(batch=2, time=90)
    shifted = x.clone()
    shifted[1] += 1000.0
    first = LarimoreStateSpace(2, 5, mode="pooled").fit(x)
    second = LarimoreStateSpace(2, 5, mode="pooled").fit(shifted)
    torch.testing.assert_close(first.transition_, second.transition_, rtol=1e-5, atol=1e-5)
