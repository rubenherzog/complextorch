import torch

from complextorch import InnovationsStateSpace, optimise_dynamical_dependence, orthonormalise_projection
from complextorch.dd_optimization import (
    optimise_dynamical_dependence_proxy_riemannian,
    optimise_dynamical_dependence_spectral_riemannian,
)


def _system():
    dtype = torch.float64
    return InnovationsStateSpace(
        torch.diag(torch.tensor([0.4, 0.3, 0.2], dtype=dtype)),
        torch.eye(3, dtype=dtype),
        0.2 * torch.eye(3, dtype=dtype),
        torch.eye(3, dtype=dtype),
    )


def test_riemannian_backends_live_in_dd_optimization_and_are_callable():
    system = _system()
    initial = orthonormalise_projection(torch.tensor([[0.6, -0.3, 0.7]], dtype=torch.float64))
    proxy = optimise_dynamical_dependence_proxy_riemannian(system, initial, max_iterations=5)
    spectral = optimise_dynamical_dependence_spectral_riemannian(
        system, initial, torch.linspace(0.0, 0.5, 9, dtype=torch.float64), max_iterations=5
    )
    assert torch.isfinite(proxy.objective).all()
    assert torch.isfinite(spectral.objective).all()


def test_public_armijo_uses_staged_ssdi_workflow():
    system = _system()
    initial = orthonormalise_projection(torch.randn((3, 1, 3), dtype=torch.float64))
    result = optimise_dynamical_dependence(
        system,
        initial,
        optimizer="armijo",
        preoptimization_max_iterations=5,
        spectral_max_iterations=5,
        frequency_points=9,
    )
    assert result.preoptimization.optimizer == "armijo"
    assert result.spectral.optimizer == "armijo"
