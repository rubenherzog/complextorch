import inspect

from complextorch import optimise_dynamical_dependence


def test_unified_dd_optimizer_defaults_to_complexbox():
    signature = inspect.signature(optimise_dynamical_dependence)
    assert signature.parameters["optimizer"].default == "complexbox"
