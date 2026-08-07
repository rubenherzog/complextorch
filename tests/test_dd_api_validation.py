import pytest
import torch

from complextorch import InnovationsStateSpace, optimise_dynamical_dependence


def _system():
    eye = torch.eye(2, dtype=torch.float64)
    return InnovationsStateSpace(0.3 * eye, eye, 0.1 * eye, eye)


def test_dispatcher_rejects_unknown_backend_options_at_backend_boundary():
    system = _system()
    initial = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    with pytest.raises(TypeError):
        optimise_dynamical_dependence(
            system,
            initial,
            objective="proxy",
            optimizer="complexbox",
            optimizer_options={"not_an_option": 1},
        )
