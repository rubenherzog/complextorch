"""Focused tests for state-space recovery, prediction, and EM refinement."""

import pytest
import torch

from complextorch import (
    LarimoreStateSpace,
    LinearGaussianEM,
    N4SID,
    StateSpaceModel,
    fit_diagnostics,
    state_space_recovery_diagnostics,
)


def _general_system():
    transition = torch.tensor(
        [[0.78, 0.12], [-0.06, 0.62]], dtype=torch.float64
    )
    observation = torch.tensor(
        [[1.0, 0.2], [0.1, 0.9], [0.4, -0.3]], dtype=torch.float64
    )
    process_covariance = torch.tensor(
        [[0.12, 0.02], [0.02, 0.08]], dtype=torch.float64
    )
    observation_covariance = torch.tensor(
        [[0.30, 0.03, 0.00], [0.03, 0.25, 0.02], [0.00, 0.02, 0.20]],
        dtype=torch.float64,
    )
    return StateSpaceModel(
        transition,
        observation,
        process_covariance,
        observation_covariance,
        state_covariance=torch.eye(2, dtype=torch.float64),
    )


def _data(batch=1, time=240, variables=3, seed=31):
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(
        batch, time, variables, generator=generator, dtype=torch.float64
    )
    values = torch.zeros_like(noise)
    transition = torch.tensor(
        [[0.62, 0.08, 0.0], [-0.04, 0.55, 0.07], [0.0, 0.03, 0.48]],
        dtype=torch.float64,
    )[:variables, :variables]
    for index in range(1, time):
        values[:, index] = values[:, index - 1] @ transition.T + 0.35 * noise[:, index]
    return values[0] if batch == 1 else values


def test_state_space_recovery_is_invariant_to_latent_basis():
    reference = _general_system()
    transform = torch.tensor([[1.4, 0.3], [-0.2, 0.9]], dtype=torch.float64)
    transform_inverse = torch.linalg.inv(transform)
    equivalent = StateSpaceModel(
        transform @ reference.transition @ transform_inverse,
        reference.observation @ transform_inverse,
        transform @ reference.process_covariance @ transform.T,
        reference.observation_covariance,
        state_covariance=transform @ reference.state_covariance @ transform.T,
    )

    result = state_space_recovery_diagnostics(equivalent, reference, horizon=4)

    assert result.spectral_distance < 1e-10
    assert result.hankel_relative_error < 1e-8
    assert result.hankel_spectrum_relative_error < 1e-8
    assert result.innovation_covariance_relative_error < 1e-8


def test_state_space_recovery_supports_batched_models():
    reference = _general_system()
    perturbed = StateSpaceModel(
        torch.stack((reference.transition, 0.9 * reference.transition)),
        reference.observation.expand(2, -1, -1).clone(),
        reference.process_covariance.expand(2, -1, -1).clone(),
        reference.observation_covariance.expand(2, -1, -1).clone(),
        state_covariance=reference.state_covariance.expand(2, -1, -1).clone(),
    )

    result = state_space_recovery_diagnostics(perturbed, reference, horizon=3)

    assert result.spectral_distance.shape == (2,)
    assert result.hankel_relative_error.shape == (2,)
    assert result.hankel_singular_values.shape[0] == 2
    assert result.spectral_distance[0] < 1e-10
    assert result.spectral_distance[1] > 1e-2


def test_fit_diagnostics_exposes_predictions_without_a_second_recursion():
    observations = _data(time=320)
    train, test = observations[:220], observations[220:]
    estimator = N4SID(2, block_rows=6, dtype="float64").fit(train)

    result = fit_diagnostics(estimator, train, test, max_lag=6)

    assert result.prediction_mean.shape == test.shape
    assert result.standardized_errors.shape == test.shape
    assert result.prediction_covariance.shape == (3, 3)
    assert 0.0 <= float(result.prediction_interval_coverage) <= 1.0
    torch.testing.assert_close(
        result.prediction_mean + (test - result.prediction_mean), test
    )


def test_predictive_extensions_preserve_independent_batch_semantics():
    observations = _data(batch=2, time=260)
    train, test = observations[:, :180], observations[:, 180:]
    estimator = N4SID(
        2, block_rows=5, mode="independent", dtype="float64"
    ).fit(train)

    result = fit_diagnostics(estimator, train, test, max_lag=5)

    assert result.prediction_mean.shape == test.shape
    assert result.standardized_errors.shape == test.shape
    assert result.prediction_covariance.shape == (2, 3, 3)
    assert result.prediction_interval_coverage.shape == (2,)


def test_n4sid_can_initialize_em_without_new_fitting_api():
    observations = _data(batch=2, time=180)
    initializer = N4SID(2, block_rows=5, mode="pooled", dtype="float64").fit(
        observations
    )
    refined = LinearGaussianEM(initializer, n_iter=2, mode="pooled").fit(
        observations
    )

    assert refined.system_.transition.shape == (2, 2)
    assert refined.mean_.shape == (2, 3)
    assert refined.trajectory_log_likelihood_history_.shape == (2, 2)
    assert refined.trajectory_initial_log_likelihood_.shape == (2,)
    assert refined.trajectory_final_log_likelihood_.shape == (2,)
    torch.testing.assert_close(
        refined.log_likelihood_gain_,
        refined.final_log_likelihood_ - refined.initial_log_likelihood_,
    )
    assert torch.isfinite(refined.log_likelihood_gain_)

    diagnostics = fit_diagnostics(
        refined, observations, evaluation="in_sample", max_lag=5
    )
    assert torch.isfinite(diagnostics.gaussian_nll)


def test_em_rejects_innovations_form_initializer_explicitly():
    observations = _data(time=160)
    initializer = LarimoreStateSpace(2, 5, dtype="float64").fit(observations)

    with pytest.raises(TypeError, match="innovations-form"):
        LinearGaussianEM(initializer, n_iter=1).fit(observations)
