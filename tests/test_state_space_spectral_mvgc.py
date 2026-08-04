import torch
from complextorch import build_var_system, var_to_innovations_state_space, reduce_innovations_state_space
from complextorch.measures import state_space_temporal_mvgc, state_space_spectral_mvgc, integrate_spectral_mvgc, pairwise_spectral_gc


def conditioned_system():
    coefficients = torch.tensor(
        [[[0.50, 0.25, 0.00], [0.00, 0.40, 0.15], [0.00, 0.00, 0.30]]],
        dtype=torch.float64,
    )
    covariance = torch.tensor(
        [[1.00, 0.20, 0.10], [0.20, 1.20, 0.15], [0.10, 0.15, 0.90]],
        dtype=torch.float64,
    )
    return build_var_system(coefficients, covariance)


def test_state_space_spectral_integrates_to_temporal_gc():
    system = conditioned_system()
    frequencies = torch.linspace(0.0, 0.5, 4097, dtype=torch.float64)
    spectral = state_space_spectral_mvgc(system, source=[1], target=[0], conditional=[2], frequencies=frequencies)
    temporal = state_space_temporal_mvgc(system, source=[1], target=[0], conditional=[2])
    integrated = integrate_spectral_mvgc(spectral, frequencies)
    torch.testing.assert_close(integrated, temporal, rtol=1e-8, atol=1e-10)
    assert torch.isfinite(spectral).all()


def test_reduced_innovations_model_is_obtained_without_refitting():
    system = conditioned_system()
    innovations = var_to_innovations_state_space(system)
    reduced = reduce_innovations_state_space(innovations, [0, 2])
    assert reduced.observation.shape[-2] == 2
    assert reduced.innovation_covariance.shape[-2:] == (2, 2)
    assert torch.linalg.eigvalsh(reduced.innovation_covariance).min() > 0


def test_bivariate_state_space_matches_specialized_geweke_curve():
    coefficients = torch.tensor([[[0.55, 0.30], [0.00, 0.35]]], dtype=torch.float64)
    covariance = torch.tensor([[1.0, 0.2], [0.2, 0.8]], dtype=torch.float64)
    system = build_var_system(coefficients, covariance)
    frequencies = torch.linspace(0.0, 0.5, 257, dtype=torch.float64)
    exact = state_space_spectral_mvgc(system, source=[1], target=[0], frequencies=frequencies)
    specialized = pairwise_spectral_gc(system, source=1, target=0, frequencies=frequencies)
    torch.testing.assert_close(exact, specialized, rtol=1e-7, atol=1e-9)
