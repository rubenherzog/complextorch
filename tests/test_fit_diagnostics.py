"""Synthetic validation for common VAR/state-space fit diagnostics."""
import torch

from complextorch import N4SID, VAR, fit_diagnostics, innovation_diagnostics
from complextorch.simulate import simulate_var


def _latent_measurement_data(n_times=1200, seed=1):
    """One latent persistent mode observed through four noisy channels."""
    generator = torch.Generator().manual_seed(seed)
    state = torch.zeros(1, dtype=torch.float64)
    values = []
    for index in range(n_times + 500):
        state = 0.98 * state + (0.05**0.5) * torch.randn(
            1, generator=generator, dtype=torch.float64
        )
        observation = state.expand(4) + (2.0**0.5) * torch.randn(
            4, generator=generator, dtype=torch.float64
        )
        if index >= 500:
            values.append(observation.clone())
    return torch.stack(values)


def test_innovation_diagnostics_detect_temporal_dependence_and_covariance_mismatch():
    generator = torch.Generator().manual_seed(12)
    n_times, n_variables = 700, 3
    white = torch.randn(
        n_times, n_variables, generator=generator, dtype=torch.float64
    )
    correlated = torch.zeros_like(white)
    forcing = torch.randn(
        n_times, n_variables, generator=generator, dtype=torch.float64
    )
    for index in range(1, n_times):
        correlated[index] = 0.8 * correlated[index - 1] + forcing[index]

    good = innovation_diagnostics(
        white,
        white,
        torch.eye(n_variables, dtype=torch.float64),
        training_mean=torch.zeros(n_variables, dtype=torch.float64),
        max_lag=8,
    )
    temporally_bad = innovation_diagnostics(
        correlated,
        correlated,
        torch.eye(n_variables, dtype=torch.float64),
        training_mean=torch.zeros(n_variables, dtype=torch.float64),
        max_lag=8,
    )
    covariance_bad = innovation_diagnostics(
        white,
        white,
        4.0 * torch.eye(n_variables, dtype=torch.float64),
        training_mean=torch.zeros(n_variables, dtype=torch.float64),
        max_lag=8,
    )

    assert good.whiteness_energy < 0.2
    assert temporally_bad.whiteness_energy > 10 * good.whiteness_energy
    assert good.covariance_calibration < 0.15
    assert covariance_bad.covariance_calibration > 5 * good.covariance_calibration
    assert bool(((good.durbin_watson > 1.7) & (good.durbin_watson < 2.3)).all())
    assert bool((temporally_bad.durbin_watson < 1.0).all())


def test_pooled_whiteness_never_connects_trajectory_boundaries():
    generator = torch.Generator().manual_seed(19)
    errors = torch.randn((4, 250, 2), generator=generator, dtype=torch.float64)
    shifted = errors + torch.tensor(
        [0.0, 20.0, -15.0, 30.0], dtype=torch.float64
    )[:, None, None]
    result = innovation_diagnostics(
        shifted,
        errors,
        torch.eye(2, dtype=torch.float64),
        training_mean=shifted.mean(1),
        max_lag=6,
        mode="pooled",
    )
    assert result.whiteness_energy < 0.15
    assert bool(((result.durbin_watson > 1.7) & (result.durbin_watson < 2.3)).all())


def test_var_and_state_space_are_both_adequate_on_simple_var1_data():
    coefficients = torch.tensor(
        [[[0.45, 0.08], [-0.03, 0.35]]], dtype=torch.float64
    )
    covariance = torch.tensor([[0.4, 0.05], [0.05, 0.3]], dtype=torch.float64)
    observations = simulate_var(
        coefficients, covariance, 1200, burnin=300, seed=8
    )[0]
    train, test = observations[:800], observations[800:]
    var = VAR(1, solver="lstsq", dtype="float64").fit(train)
    state_space = N4SID(2, 5, dtype="float64").fit(train)

    var_diagnostics = fit_diagnostics(var, train, test, max_lag=8)
    ss_diagnostics = fit_diagnostics(state_space, train, test, max_lag=8)

    assert var_diagnostics.whiteness_energy < 0.15
    assert ss_diagnostics.whiteness_energy < 0.15
    assert var_diagnostics.predictive_r2 > 0.1
    assert ss_diagnostics.predictive_r2 > 0.1


def test_correct_var_beats_underfit_state_space_on_var2_data():
    coefficients = torch.zeros((2, 2, 2), dtype=torch.float64)
    coefficients[0] = torch.tensor(
        [[0.55, 0.12], [-0.05, 0.45]], dtype=torch.float64
    )
    coefficients[1] = torch.tensor(
        [[0.18, 0.0], [0.08, 0.10]], dtype=torch.float64
    )
    covariance = torch.tensor(
        [[0.25, 0.04], [0.04, 0.20]], dtype=torch.float64
    )
    observations = simulate_var(
        coefficients, covariance, 1000, burnin=300, seed=2
    )[0]
    train, test = observations[:700], observations[700:]
    var = VAR(2, solver="lstsq", dtype="float64").fit(train)
    state_space = N4SID(1, 6, dtype="float64").fit(train)

    var_diagnostics = fit_diagnostics(var, train, test, max_lag=8)
    ss_diagnostics = fit_diagnostics(state_space, train, test, max_lag=8)

    assert var_diagnostics.gaussian_nll < ss_diagnostics.gaussian_nll
    assert var_diagnostics.predictive_r2 > ss_diagnostics.predictive_r2
    assert var_diagnostics.whiteness_energy < 0.25 * ss_diagnostics.whiteness_energy


def test_state_space_beats_underfit_var_on_latent_measurement_data():
    observations = _latent_measurement_data()
    train, test = observations[:780], observations[780:]
    state_space = N4SID(1, 12, dtype="float64").fit(train)
    var = VAR(1, solver="lstsq", dtype="float64").fit(train)

    ss_diagnostics = fit_diagnostics(state_space, train, test, max_lag=8)
    var_diagnostics = fit_diagnostics(var, train, test, max_lag=8)

    assert ss_diagnostics.gaussian_nll < var_diagnostics.gaussian_nll
    assert ss_diagnostics.predictive_r2 > var_diagnostics.predictive_r2
    assert ss_diagnostics.whiteness_energy < var_diagnostics.whiteness_energy


def test_fit_diagnostics_rejects_invalid_holdout_requests():
    data = torch.randn((80, 2), dtype=torch.float64)
    estimator = VAR(2, solver="lstsq", dtype="float64").fit(data[:60])
    try:
        fit_diagnostics(estimator, data[:60], data[60:], max_lag=20)
    except ValueError as error:
        assert "max_lag" in str(error)
    else:
        raise AssertionError("max_lag >= n_test must fail")


def test_independent_batch_preserves_one_diagnostic_per_trajectory():
    coefficients = torch.stack(
        [
            torch.tensor([[[0.35, 0.02], [0.01, 0.30]]], dtype=torch.float64),
            torch.tensor([[[0.55, -0.04], [0.06, 0.40]]], dtype=torch.float64),
        ]
    )
    covariance = torch.eye(2, dtype=torch.float64).expand(2, -1, -1).clone() * 0.3
    observations = simulate_var(
        coefficients, covariance, 700, burnin=250, seed=23
    )
    train, test = observations[:, :500], observations[:, 500:]
    estimator = VAR(
        1, mode="independent", solver="lstsq", dtype="float64"
    ).fit(train)
    result = fit_diagnostics(estimator, train, test, max_lag=6)

    assert result.rmse.shape == (2,)
    assert result.whiteness_energy.shape == (2,)
    assert result.innovation_covariance_oos.shape == (2, 2, 2)
    assert result.autocorrelation_matrices.shape == (2, 6, 2, 2)
