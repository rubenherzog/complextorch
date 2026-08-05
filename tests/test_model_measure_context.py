import numpy as np
import torch
from scipy import stats as scipy_stats

from complextorch import (
    ModelMeasureConfig, VAR, VAROrderSelectionIC, automatic_burnin,
    build_measure_context, build_var_system, demo_var, model_autocovariances,
    mvgc_pvalue, phiid_from_model, random_correlation_matrix,
    significance, simulate_var, spectral_mvgc, temporal_mvgc,
)
from complextorch.measures import integrate_spectral_mvgc


def test_var_and_equivalent_state_space_share_autocovariances_and_phiid():
    coefficients, innovation_covariance = demo_var(n_variables=3, order=2)
    system = build_var_system(coefficients, innovation_covariance)
    state_space = system.to_state_space()
    gamma_var = model_autocovariances(system, 4)
    gamma_state = model_autocovariances(state_space, 4)
    torch.testing.assert_close(gamma_state, gamma_var, rtol=1e-9, atol=1e-11)
    phiid_var = phiid_from_model(system, (0, 1), lag=3, autocovariance_sequence=gamma_var)
    phiid_state = phiid_from_model(state_space, (0, 1), lag=3, autocovariance_sequence=gamma_state)
    for name in phiid_var:
        torch.testing.assert_close(phiid_state[name], phiid_var[name], rtol=1e-8, atol=1e-10)


def test_independent_measure_lags_resolve_to_single_maximum():
    coefficients, innovation_covariance = demo_var(n_variables=3, order=2)
    system = build_var_system(coefficients, innovation_covariance)
    config = ModelMeasureConfig(autocovariance_max_lag=2, cmem_max_lag=7, phiid_variables=(0, 1), phiid_lag=5)
    context = build_measure_context(system, config)
    assert context.max_lag == 7
    assert context.autocovariances.shape[1] == 8


def test_primary_mvgc_is_model_based_and_spectral_integrates_to_temporal():
    coefficients = torch.tensor([[[0.50, 0.25, 0.00], [0.00, 0.40, 0.15], [0.00, 0.00, 0.30]]], dtype=torch.float64)
    covariance = torch.tensor([[1.00, 0.20, 0.10], [0.20, 1.20, 0.15], [0.10, 0.15, 0.90]], dtype=torch.float64)
    system = build_var_system(coefficients, covariance)
    frequencies = torch.linspace(0.0, 0.5, 4097, dtype=torch.float64)
    temporal = temporal_mvgc(system, source=(1,), target=(0,), conditional=(2,))
    spectral = spectral_mvgc(system, source=(1,), target=(0,), conditional=(2,), frequencies=frequencies)
    integrated = integrate_spectral_mvgc(spectral, frequencies)
    torch.testing.assert_close(integrated, temporal, rtol=1e-8, atol=1e-10)


def test_onion_correlation_and_simulation_options():
    correlation = random_correlation_matrix(5, batch=3, seed=12)
    torch.testing.assert_close(torch.diagonal(correlation, dim1=-2, dim2=-1), torch.ones(3, 5))
    assert bool((torch.linalg.eigvalsh(correlation) > 0).all())
    coefficients, covariance = demo_var(n_variables=3, order=2)
    assert automatic_burnin(coefficients) > 0
    observations, innovations = simulate_var(coefficients, covariance, 300, burnin='auto', seed=5, return_innovations=True)
    assert observations.shape == innovations.shape == (1, 300, 3)


def test_lwr_solver_diagnostics_and_information_criteria():
    coefficients, covariance = demo_var(n_variables=3, order=2)
    observations = simulate_var(coefficients, covariance, 3500, burnin='auto', seed=7)
    lwr = VAR(order=2, solver='lwr', covariance='unbiased', device='cpu').fit(observations)
    ols = VAR(order=2, solver='lstsq', covariance='unbiased', device='cpu').fit(observations)
    assert lwr.params_.solver == 'lwr'
    assert bool((lwr.spectral_radius_ < 1).all())
    torch.testing.assert_close(lwr.coef_, ols.coef_, rtol=0.15, atol=0.08)
    assert lwr.consistency(observations) > 0.8
    whiteness = lwr.whiteness(observations, method='durbin_watson')
    assert whiteness.statistic.shape == whiteness.pvalue.shape == (3,)
    selector = VAROrderSelectionIC(range(1, 5), solver='lwr', refit='hqc', device='cpu').fit(observations)
    assert selector.p_hqc_ in range(1, 5)
    assert selector.aic_.shape == selector.bic_.shape == selector.hqc_.shape == (4,)


def test_mvgc_statistics_match_reference_formula_and_bh():
    value = torch.tensor(0.12)
    pvalue = mvgc_pvalue(value, method='F', n_target=1, n_source=1, n_conditional=3, order=2, n_times=1000, n_trials=1)
    d = 2
    d2 = 1 * ((1000 - 2) - 2 * 5)
    expected = scipy_stats.f.sf((d2 / d) * 0.12, d, d2)
    assert np.isclose(float(pvalue), expected)
    pvalues = torch.tensor([0.001, 0.01, 0.03, 0.20, float('nan')])
    mask = significance(pvalues, alpha=0.05, method='FDR')
    assert mask.tolist() == [True, True, True, False, False]
