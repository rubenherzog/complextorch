import torch

from complextorch import (
    ModelMeasureConfig,
    build_measure_context,
    build_var_system,
    demo_var,
    model_autocovariances,
    phiid_from_model,
    spectral_mvgc,
    temporal_mvgc,
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
    config = ModelMeasureConfig(
        autocovariance_max_lag=2,
        cmem_max_lag=7,
        phiid_variables=(0, 1),
        phiid_lag=5,
    )
    context = build_measure_context(system, config)
    assert context.max_lag == 7
    assert context.autocovariances.shape[1] == 8


def test_primary_mvgc_is_model_based_and_spectral_integrates_to_temporal():
    coefficients = torch.tensor(
        [[[0.50, 0.25, 0.00], [0.00, 0.40, 0.15], [0.00, 0.00, 0.30]]],
        dtype=torch.float64,
    )
    covariance = torch.tensor(
        [[1.00, 0.20, 0.10], [0.20, 1.20, 0.15], [0.10, 0.15, 0.90]],
        dtype=torch.float64,
    )
    system = build_var_system(coefficients, covariance)
    frequencies = torch.linspace(0.0, 0.5, 4097, dtype=torch.float64)

    temporal = temporal_mvgc(system, source=(1,), target=(0,), conditional=(2,))
    spectral = spectral_mvgc(
        system,
        source=(1,),
        target=(0,),
        conditional=(2,),
        frequencies=frequencies,
    )
    integrated = integrate_spectral_mvgc(spectral, frequencies)
    torch.testing.assert_close(integrated, temporal, rtol=1e-8, atol=1e-10)
