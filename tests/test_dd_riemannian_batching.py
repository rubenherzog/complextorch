import torch

from complextorch import InnovationsStateSpace, optimise_dynamical_dependence, orthonormalise_projection


def test_armijo_restart_batch_remains_independent():
    dtype = torch.float64
    system = InnovationsStateSpace(
        torch.diag(torch.tensor([0.4, 0.3, 0.2], dtype=dtype)),
        torch.eye(3, dtype=dtype),
        0.2 * torch.eye(3, dtype=dtype),
        torch.eye(3, dtype=dtype),
    )
    initial = orthonormalise_projection(torch.randn((4, 1, 3), dtype=dtype))
    result = optimise_dynamical_dependence(
        system,
        initial,
        optimizer="armijo",
        preoptimization_max_iterations=5,
        spectral_max_iterations=5,
        frequency_points=9,
    )
    assert result.preoptimization.projection.shape[0] == 4
    assert int(result.cluster_sizes.sum()) == 4
    assert torch.isfinite(result.objective).all()
