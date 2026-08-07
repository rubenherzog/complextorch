import numpy as np
import torch
from scipy.linalg import cholesky

from complextorch import LarimoreStateSpace


def _reference_larimore(observations, past_horizon, future_horizon, n_states, ridge=1e-12):
    """Minimal output-only CVA implementation from Larimore/Bauer equations."""
    values = np.asarray(observations, dtype=float)
    if values.ndim == 2:
        values = values[None]
    values = values - values.mean(axis=1, keepdims=True)
    n_batch, n_times, n_outputs = values.shape
    columns = n_times - past_horizon - future_horizon + 1

    past_trials = []
    future_trials = []
    for batch in range(n_batch):
        past = np.column_stack([
            np.concatenate([
                values[batch, time - 1 - lag]
                for lag in range(past_horizon)
            ])
            for time in range(past_horizon, n_times - future_horizon + 1)
        ])
        future = np.column_stack([
            np.concatenate([
                values[batch, time + lag]
                for lag in range(future_horizon)
            ])
            for time in range(past_horizon, n_times - future_horizon + 1)
        ])
        past_trials.append(past)
        future_trials.append(future)

    past = np.concatenate(past_trials, axis=1)
    future = np.concatenate(future_trials, axis=1)
    n_effective = past.shape[1]
    spp = past @ past.T / n_effective
    sff = future @ future.T / n_effective
    spf = past @ future.T / n_effective
    lp = cholesky(spp + ridge * np.eye(spp.shape[0]), lower=True)
    lf = cholesky(sff + ridge * np.eye(sff.shape[0]), lower=True)
    whitened = np.linalg.solve(lf, spf.T) @ np.linalg.inv(lp.T)
    _, correlations, right = np.linalg.svd(whitened, full_matrices=False)

    flat_states = (
        np.diag(correlations[:n_states])
        @ right[:n_states]
        @ np.linalg.inv(lp).T
        @ past
    )
    states = flat_states.T.reshape(n_batch, columns, n_states)
    observations_all = values[:, past_horizon : past_horizon + columns]

    state_all = states.reshape(-1, n_states)
    output_all = observations_all.reshape(-1, n_outputs)
    observation = np.linalg.lstsq(state_all, output_all, rcond=None)[0].T
    innovations = observations_all - states @ observation.T
    innovations_flat = innovations.reshape(-1, n_outputs)
    innovation_covariance = innovations_flat.T @ innovations_flat / innovations_flat.shape[0]

    previous = states[:, :-1].reshape(-1, n_states)
    following = states[:, 1:].reshape(-1, n_states)
    transition = np.linalg.lstsq(previous, following, rcond=None)[0].T
    state_residual = following - previous @ transition.T
    transition_innovations = innovations[:, :-1].reshape(-1, n_outputs)
    gain = np.linalg.lstsq(transition_innovations, state_residual, rcond=None)[0].T
    return transition, observation, gain, innovation_covariance, correlations


def _data(seed=11, n_batch=3, n_times=160):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n_batch, n_times, 3, generator=generator, dtype=torch.float64)


def test_larimore_matches_equation_level_reference_for_pooled_trials():
    observations = _data()
    expected = _reference_larimore(observations.numpy(), 5, 4, 2)
    fitted = LarimoreStateSpace(
        2, 5, future_horizon=4, mode="pooled", covariance="mle"
    ).fit(observations)

    torch.testing.assert_close(fitted.transition_, torch.from_numpy(expected[0]), atol=2e-10, rtol=2e-10)
    torch.testing.assert_close(fitted.observation_, torch.from_numpy(expected[1]), atol=2e-10, rtol=2e-10)
    torch.testing.assert_close(fitted.kalman_gain_, torch.from_numpy(expected[2]), atol=2e-10, rtol=2e-10)
    torch.testing.assert_close(fitted.innovation_covariance_, torch.from_numpy(expected[3]), atol=2e-10, rtol=2e-10)
    torch.testing.assert_close(fitted.canonical_correlations_, torch.from_numpy(expected[4]), atol=2e-10, rtol=2e-10)


def test_larimore_pooled_fit_is_invariant_to_trial_order():
    observations = _data(seed=29, n_batch=5)
    first = LarimoreStateSpace(2, 5, mode="pooled").fit(observations)
    second = LarimoreStateSpace(2, 5, mode="pooled").fit(observations.flip(0))

    torch.testing.assert_close(first.transition_, second.transition_, atol=3e-10, rtol=3e-10)
    torch.testing.assert_close(first.observation_, second.observation_, atol=3e-10, rtol=3e-10)
    torch.testing.assert_close(first.kalman_gain_, second.kalman_gain_, atol=3e-10, rtol=3e-10)
    torch.testing.assert_close(first.innovation_covariance_, second.innovation_covariance_, atol=3e-10, rtol=3e-10)


def test_larimore_mle_covariance_is_sample_moment_of_all_innovations():
    observations = _data(seed=37, n_batch=2)
    fitted = LarimoreStateSpace(2, 5, mode="pooled").fit(observations)
    innovations = fitted.innovations_.reshape(-1, observations.shape[-1])
    expected = innovations.T @ innovations / innovations.shape[0]
    torch.testing.assert_close(fitted.innovation_covariance_, expected + 1e-12 * torch.eye(3, dtype=expected.dtype), atol=2e-12, rtol=2e-12)
