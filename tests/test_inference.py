import pytest
import torch

from complextorch import (
    InferenceMeasureConfig,
    LarimoreStateSpace,
    LinearGaussianEM,
    ModelMeasureConfig,
    N4SID,
    VAR,
    simulate_var,
)
from complextorch._resampling import _batched_ols_refit
from complextorch.inference import NuMITPIDResult, measure_confidence_intervals, numit_pid_var


def test_inference_namespace_exposes_numit_separately_from_resampling():
    """NuMIT is public inference API without becoming a ResamplingMethod."""
    from complextorch import NuMITPIDResult as TopLevelNuMITPIDResult
    from complextorch import numit_pid_var as top_level_numit_pid_var

    assert NuMITPIDResult is TopLevelNuMITPIDResult
    assert numit_pid_var is top_level_numit_pid_var


def _stable_data(*, batch=2, n_times=180, seed=7, dtype=torch.float64):
    base = torch.tensor(
        [[[0.42, 0.08], [-0.05, 0.31]]], dtype=dtype
    )
    coefficients = base.expand(batch, -1, -1, -1).contiguous()
    covariance = torch.tensor(
        [[1.0, 0.25], [0.25, 0.8]], dtype=dtype
    ).expand(batch, -1, -1).contiguous()
    return simulate_var(coefficients, covariance, n_times, burnin=150, seed=seed)


def test_pooled_refit_never_constructs_cross_trial_lags():
    estimator = VAR(order=1, mode="pooled", fit_intercept=True, covariance="mle")
    samples = torch.tensor(
        [[[[0.0], [1.0], [2.0], [3.0]], [[100.0], [101.0], [102.0], [103.0]]]],
        dtype=torch.float64,
    )
    coefficients, covariance = _batched_ols_refit(samples, estimator)
    torch.testing.assert_close(coefficients[0, 0, 0, 0], torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(covariance, torch.zeros_like(covariance), atol=1e-24, rtol=0.0)


@pytest.mark.parametrize("method", ["residual_bootstrap", "parametric"])
def test_pooled_confidence_intervals_are_reproducible_and_reuse_one_ensemble(method):
    data = _stable_data()
    kwargs = dict(
        measures=["o_information", "spectral_radius"],
        var=VAR(order=1, mode="pooled", dtype="float64"),
        method=method,
        n_resamples=24,
        confidence=0.9,
        seed=123,
        return_samples=True,
    )
    first = measure_confidence_intervals(data, **kwargs)
    second = measure_confidence_intervals(data, **kwargs)
    assert set(first.intervals) == {
        "gaussian.o_information",
        "criticality.spectral_radius",
    }
    assert first.n_valid + first.n_failed == 24
    assert first.n_valid >= 2
    assert first["gaussian.o_information"].samples.shape[0] == first.n_valid
    assert first["criticality.spectral_radius"].samples.shape[0] == first.n_valid
    for name in first.intervals:
        torch.testing.assert_close(first[name].samples, second[name].samples)
        torch.testing.assert_close(first[name].lower, second[name].lower)
        torch.testing.assert_close(first[name].upper, second[name].upper)


def test_independent_mode_preserves_trajectory_axis_in_intervals():
    data = _stable_data(batch=2, seed=13)
    result = measure_confidence_intervals(
        data,
        measures="o_information",
        var=VAR(order=1, mode="independent", dtype="float64"),
        n_resamples=20,
        seed=9,
        return_samples=True,
    )
    interval = result["gaussian.o_information"]
    assert interval.estimate.shape == (2,)
    assert interval.lower.shape == (2,)
    assert interval.upper.shape == (2,)
    assert interval.samples.shape == (result.n_valid, 2)


def test_dynamical_dependence_uses_fixed_projection_batch_without_optimization():
    data = _stable_data(batch=2, seed=21)
    projections = torch.tensor(
        [[[1.0, 0.0]], [[0.0, 1.0]]], dtype=torch.float64
    )
    result = measure_confidence_intervals(
        data,
        measures="dynamical_dependence",
        var=VAR(order=1, mode="independent", dtype="float64"),
        config=InferenceMeasureConfig(
            primary=ModelMeasureConfig(macro_projection=projections)
        ),
        n_resamples=16,
        seed=4,
        return_samples=True,
    )
    interval = result["control.dynamical_dependence"]
    assert interval.estimate.shape == (2,)
    assert interval.samples.shape == (result.n_valid, 2)


def test_invalid_resampling_requests_fail_explicitly():
    data = _stable_data(batch=1, n_times=100)
    with pytest.raises(ValueError, match="n_resamples"):
        measure_confidence_intervals(data, n_resamples=1)
    with pytest.raises(ValueError, match="confidence"):
        measure_confidence_intervals(data, n_resamples=4, confidence=1.0)
    with pytest.raises(ValueError, match="unregularized"):
        measure_confidence_intervals(
            data,
            var=VAR(order=1, mode="pooled", alpha=0.1),
            n_resamples=4,
        )
    with pytest.raises(ValueError, match="lwr"):
        measure_confidence_intervals(
            data,
            var=VAR(order=1, mode="pooled", solver="lwr"),
            n_resamples=4,
        )



def test_state_space_parametric_ci_converges_with_var_and_innovations():
    """Equivalent model families give compatible finite-sample PI intervals."""
    coefficients = torch.tensor(
        [[[0.45, 0.10], [-0.05, 0.30]]], dtype=torch.float64
    )
    covariance = torch.tensor(
        [[0.9, 0.18], [0.18, 0.7]], dtype=torch.float64
    )
    data = simulate_var(coefficients, covariance, 800, burnin=250, seed=2826)[0]
    initial = N4SID(2, block_rows=6, mode="pooled", dtype="float64").fit(data).system_
    estimators = (
        VAR(order=1, mode="pooled", dtype="float64"),
        LinearGaussianEM(initial, n_iter=5, mode="pooled"),
        LarimoreStateSpace(
            2, 6, future_horizon=6, mode="pooled", dtype="float64"
        ),
    )
    intervals = []
    for estimator in estimators:
        result = measure_confidence_intervals(
            data,
            measures="predictive_information",
            estimator=estimator,
            method="parametric",
            n_resamples=24,
            confidence=0.95,
            seed=123,
        )
        interval = result["dynamics.predictive_information"]
        intervals.append(interval)
        assert result.n_valid == 24
        assert bool(interval.lower <= interval.estimate <= interval.upper)

    points = torch.stack([interval.estimate for interval in intervals])
    lowers = torch.stack([interval.lower for interval in intervals])
    uppers = torch.stack([interval.upper for interval in intervals])
    assert bool(points.max() - points.min() < 0.03)
    assert bool(lowers.max() <= uppers.min())


def test_innovations_state_space_residual_bootstrap_is_supported():
    """Residual-vector bootstrap works directly with innovations-form fits."""
    data = _stable_data(batch=1, n_times=220, seed=31)[0]
    result = measure_confidence_intervals(
        data,
        measures="predictive_information",
        estimator=LarimoreStateSpace(
            2, 5, future_horizon=5, mode="pooled", dtype="float64"
        ),
        method="residual_bootstrap",
        n_resamples=12,
        seed=5,
        return_samples=True,
    )
    interval = result["dynamics.predictive_information"]
    assert result.estimator_type == "LarimoreStateSpace"
    assert result.var_order is None
    assert interval.samples.shape == (result.n_valid,)
    assert bool(torch.isfinite(interval.samples).all())


def test_state_space_independent_mode_preserves_trajectory_axis():
    """State-space resampling never mixes independent trajectory fits."""
    coefficients = torch.tensor(
        [
            [[[0.45, 0.10], [-0.05, 0.30]]],
            [[[0.35, -0.05], [0.08, 0.25]]],
        ],
        dtype=torch.float64,
    )
    covariance = torch.eye(2, dtype=torch.float64).expand(2, -1, -1).contiguous()
    data = simulate_var(coefficients, covariance, 260, burnin=100, seed=99)
    result = measure_confidence_intervals(
        data,
        measures="predictive_information",
        estimator=LarimoreStateSpace(2, 5, mode="independent", dtype="float64"),
        method="parametric",
        n_resamples=8,
        seed=1,
        return_samples=True,
    )
    interval = result["dynamics.predictive_information"]
    assert interval.estimate.shape == (2,)
    assert interval.lower.shape == (2,)
    assert interval.upper.shape == (2,)
    assert interval.samples.shape == (result.n_valid, 2)
