import numpy as np
import pytest
import torch

from complextorch.selection import (
    SelectionCandidate,
    gaussian_log_likelihood,
    innovations_state_space_parameter_count,
    score_information_criteria,
    select_by_information_criterion,
    symmetric_covariance_parameter_count,
    var_parameter_count,
)


def test_gaussian_log_likelihood_matches_manual_scalar_formula():
    errors = torch.tensor([[1.0, -1.0], [0.0, 2.0]], dtype=torch.float64)
    covariance = torch.eye(2, dtype=torch.float64)

    result = gaussian_log_likelihood(errors, covariance, reduction="sum")
    expected = -0.5 * (
        errors.square().sum().item()
        + errors.shape[0] * errors.shape[1] * np.log(2.0 * np.pi)
    )

    assert result.item() == pytest.approx(expected)


def test_gaussian_log_likelihood_preserves_independent_batch_semantics():
    errors = torch.zeros(1000, 5, 2, dtype=torch.float64)
    covariance = torch.eye(2, dtype=torch.float64)

    result = gaussian_log_likelihood(errors, covariance, reduction="sum")

    assert result.shape == (1000,)
    expected = -0.5 * 5 * 2 * np.log(2.0 * np.pi)
    torch.testing.assert_close(result, torch.full_like(result, expected))


def test_information_criteria_explicit_likelihood_and_output_scales_agree():
    total = score_information_criteria(-60.0, 3, 20)
    mean = score_information_criteria(
        -3.0, 3, 20, likelihood="mean", scale="total"
    )
    per_observation = score_information_criteria(
        -60.0, 3, 20, likelihood="total", scale="per_observation"
    )

    np.testing.assert_allclose(mean.aic, total.aic)
    np.testing.assert_allclose(mean.bic, total.bic)
    np.testing.assert_allclose(mean.hqc, total.hqc)
    np.testing.assert_allclose(per_observation.aic, total.aic / 20)
    np.testing.assert_allclose(per_observation.bic, total.bic / 20)
    np.testing.assert_allclose(per_observation.hqc, total.hqc / 20)


def test_selection_ranks_var_candidates():
    candidates = [
        SelectionCandidate("VAR(1)", -100.0, var_parameter_count(3, 1), 200),
        SelectionCandidate("VAR(2)", -99.5, var_parameter_count(3, 2), 200),
    ]

    result = select_by_information_criterion(candidates, criterion="bic")

    assert result.best_candidate_name == "VAR(1)"
    assert result.best_index == 0
    assert result.deltas.shape == (2,)
    assert result.deltas[0] == pytest.approx(0.0)


def test_selection_ranks_state_space_candidates():
    candidates = [
        SelectionCandidate(
            "SSM(1)", -120.0, innovations_state_space_parameter_count(4, 1), 300
        ),
        SelectionCandidate(
            "SSM(2)", -119.8, innovations_state_space_parameter_count(4, 2), 300
        ),
    ]

    result = select_by_information_criterion(candidates, criterion="hqc")

    assert result.best_candidate_name == "SSM(1)"


def test_selection_ranks_var_against_state_space_candidate():
    candidates = [
        SelectionCandidate("VAR(2)", -120.0, var_parameter_count(4, 2), 250),
        SelectionCandidate(
            "SSM(2)", -120.0, innovations_state_space_parameter_count(4, 2), 250
        ),
    ]

    result = select_by_information_criterion(candidates, criterion="aic")

    assert result.best_candidate_name == "SSM(2)"


def test_selection_supports_batched_and_broadcast_candidate_metadata():
    candidates = [
        SelectionCandidate(
            "VAR",
            log_likelihood=np.array([-20.0, -20.0]),
            n_parameters=np.array([10, 2]),
            n_observations=30,
        ),
        SelectionCandidate(
            "SSM",
            log_likelihood=np.array([-22.0, -25.0]),
            n_parameters=1,
            n_observations=np.array([30, 30]),
        ),
    ]

    result = select_by_information_criterion(candidates, criterion="aic")

    assert result.best_candidate_name.tolist() == ["SSM", "VAR"]
    assert result.scores.shape == (2, 2)
    assert result.information_criteria.bic.shape == (2, 2)


def test_selection_compares_thousand_var_ssm_pairs_without_python_dataset_loop():
    batch = 1000
    n_observations = np.full(batch, 200)
    candidates = [
        SelectionCandidate(
            "VAR",
            log_likelihood=np.linspace(-120.0, -100.0, batch),
            n_parameters=30,
            n_observations=n_observations,
        ),
        SelectionCandidate(
            "SSM",
            log_likelihood=np.linspace(-121.0, -98.0, batch),
            n_parameters=20,
            n_observations=n_observations,
        ),
    ]

    result = select_by_information_criterion(candidates, criterion="bic")

    assert result.scores.shape == (batch, 2)
    assert result.best_candidate_name.shape == (batch,)


def test_selection_rejects_mismatched_observation_windows():
    candidates = [
        SelectionCandidate("VAR", -100.0, 10, 200),
        SelectionCandidate("SSM", -100.0, 8, 199),
    ]

    with pytest.raises(ValueError, match="same n_observations"):
        select_by_information_criterion(candidates)


def test_minimal_innovations_ssm_parameter_count_quotients_state_basis():
    n, r = 4, 3
    covariance = symmetric_covariance_parameter_count(n)
    raw = r * r + 2 * n * r + covariance
    identifiable = 2 * n * r + covariance

    assert innovations_state_space_parameter_count(n, r, minimal=False) == raw
    assert innovations_state_space_parameter_count(n, r, minimal=True) == identifiable
    assert raw - identifiable == r * r


def test_existing_var_mvgc_per_observation_convention_is_reproduced():
    loglik = np.array([-2.0, -1.5])
    parameters = np.array([4.0, 8.0])
    observations = np.array([100.0, 90.0])
    result = score_information_criteria(
        loglik,
        parameters,
        observations,
        likelihood="mean",
        scale="per_observation",
    )

    np.testing.assert_allclose(result.aic, -2 * loglik + 2 * parameters / observations)
    np.testing.assert_allclose(
        result.bic, -2 * loglik + parameters * np.log(observations) / observations
    )
    np.testing.assert_allclose(
        result.hqc,
        -2 * loglik
        + 2 * parameters * np.log(np.log(observations)) / observations,
    )
