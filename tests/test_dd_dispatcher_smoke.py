import torch

from complextorch import InnovationsStateSpace, optimise_dynamical_dependence


def test_dispatcher_complexbox_smoke():
    eye = torch.eye(2, dtype=torch.float64)
    system = InnovationsStateSpace(0.25 * eye, eye, 0.1 * eye, eye)
    projection = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    result = optimise_dynamical_dependence(
        system,
        projection,
        objective="proxy",
        max_iterations=3,
    )
    assert result.optimizer == "complexbox"
    assert result.objective.shape == (1,)
