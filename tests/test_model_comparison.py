import numpy as np
import pytest

from complextorch.selection import (
    ModelSelectionCandidate,
    compare_model_candidates,
    information_criteria,
    information_criteria_from_negative_log_likelihood,
    innovations_state_space_parameter_count,
    symmetric_covariance_parameter_count,
    var_parameter_count,
)


def test_information_criteria_match_standard_total_likelihood_formulas():
    result = information_criteria(
        log_likelihood=-100.0, n_parameters=5, n_observations=50
    )

    assert result.aic == pytest.approx(210.0)
    assert result.bic == pytest.approx(200.0 + 5.0 * np.log(50.0))
    assert result.hqc == pytest.approx(200.0 + 10.0 * np.log(np.log(50.0)))


def test_information_criteria_from_mean_negative_log_likelihood():
    direct = information_criteria(
        log_likelihood=-60.0, n_parameters=3, n_observations=20
    )
    from_mean = information_criteria_from_negative_log_likelihood(
        3.0, n_parameters=3, n_observations=20, mean=True
    )

    np.testing.assert_allclose(from_mean.aic, direct.aic)
    np.testing.assert_allclose(from_mean.bic, direct.bic)
    np.testing.assert_allclose(from_mean.hqc, direct.hqc)


def test_var_and_minimal_innovations_state_space_parameter_counts():
    assert symmetric_covariance_parameter_count(4) == 10
    assert var_parameter_count(4, 2) == 4 * 4 * 2 + 10
    assert var_parameter_count(4, 2, include_intercept=True) == 4 * 4 * 2 + 4 + 10
    assert innovations_state_space_parameter_count(4, 3) == 2 * 4 * 3 + 10
    assert innovations_state_space_parameter_count(4, 3, minimal=False) == 3 * 3 + 2 * 4 * 3 + 10


def test_compare_model_candidates_selects_parsimonious_model_when_likelihoods_match():
    var = ModelSelectionCandidate(
        "VAR", log_likelihood=-120.0, n_parameters=30, n_observations=80
    )
    ssm = ModelSelectionCandidate(
        "SSM", log_likelihood=-120.0, n_parameters=20, n_observations=80
    )

    result = compare_model_candidates([var, ssm])

    assert result.best_aic == "SSM"
    assert result.best_bic == "SSM"
    assert result.best_hqc == "SSM"
    assert result.delta_aic.shape == (2,)
    assert result.delta_aic[1] == 0.0
    assert result.delta_aic[0] > 0.0


def test_compare_model_candidates_supports_batched_arrays():
    var = ModelSelectionCandidate(
        "VAR",
        log_likelihood=np.array([-20.0, -20.0]),
        n_parameters=np.array([10, 2]),
        n_observations=np.array([30, 30]),
    )
    ssm = ModelSelectionCandidate(
        "SSM",
        log_likelihood=np.array([-22.0, -25.0]),
        n_parameters=np.array([1, 1]),
        n_observations=np.array([30, 30]),
    )

    result = compare_model_candidates([var, ssm])

    assert result.best_aic.tolist() == ["SSM", "VAR"]
    assert result.aic.shape == (2, 2)


def test_compare_model_candidates_requires_unique_names():
    candidates = [
        ModelSelectionCandidate("M", -1.0, 1, 10),
        ModelSelectionCandidate("M", -2.0, 1, 10),
    ]

    with pytest.raises(ValueError, match="unique"):
        compare_model_candidates(candidates)
