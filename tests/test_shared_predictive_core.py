import torch
from complextorch import LinearDynamicalSystem, optimise_dynamical_dependence_projection
from complextorch.simulate import random_stable_var, simulate_var
from complextorch.measures import temporal_mvgc, spectral_mvgc, gaussian_phiid_atoms


def test_group_mvgc_uses_shared_full_reduced_models():
    coefficients, noise = random_stable_var(1, 4, 2, seed=40)
    observations = simulate_var(coefficients, noise, 1800, burnin=500, seed=41)
    temporal = temporal_mvgc(observations, 2, source=[1, 2], target=[0], conditional=[3])
    spectral = spectral_mvgc(observations, 2, source=[1, 2], target=[0], conditional=[3], frequencies=torch.linspace(0, 0.5, 64, dtype=torch.float64))
    assert torch.isfinite(temporal).all()
    assert spectral.shape[-1] == 64
    assert torch.isfinite(spectral).all()


def test_complete_phiid_lattice_reconstructs_total_information():
    covariance = torch.tensor([[1.0, 0.15, 0.30, 0.10], [0.15, 1.0, 0.05, 0.25], [0.30, 0.05, 1.0, 0.20], [0.10, 0.25, 0.20, 1.0]], dtype=torch.float64)
    result = gaussian_phiid_atoms(covariance)
    assert len([key for key in result if "_to_" in key]) == 16
    torch.testing.assert_close(result["reconstruction"], result["total"], rtol=1e-9, atol=1e-9)


def test_projection_search_reuses_state_space_control_pipeline():
    transition = torch.tensor([[0.8, 0.1], [0.0, 0.65]], dtype=torch.float64)
    observation = torch.tensor([[1.0, 0.0], [0.2, 1.0]], dtype=torch.float64)
    process_noise = 0.05 * torch.eye(2, dtype=torch.float64)
    observation_noise = 0.1 * torch.eye(2, dtype=torch.float64)
    system = LinearDynamicalSystem(transition, observation, process_noise, observation_noise, state_covariance=torch.eye(2, dtype=torch.float64))
    result = optimise_dynamical_dependence_projection(system, 1, n_candidates=16, seed=5)
    assert result.projection.shape == (1, 2)
    assert result.history.shape[0] == 16
