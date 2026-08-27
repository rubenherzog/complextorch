import complextorch
import complextorch.design as design
import complextorch.mechanisms as mechanisms


def test_mechanisms_module_declares_public_namespace():
    assert set(mechanisms.__all__) == {
        "ModalDecomposition",
        "modal_decomposition",
        "modal_observation_covariance",
    }


def test_design_module_and_top_level_aliases_are_consistent():
    for name in design.__all__:
        assert getattr(complextorch, name) is getattr(design, name)


def test_mechanism_top_level_aliases_are_consistent():
    for name in mechanisms.__all__:
        assert getattr(complextorch, name) is getattr(mechanisms, name)


def test_projected_innovations_is_canonical_top_level_transformation():
    from complextorch.transformations import project_innovations_state_space

    assert complextorch.project_innovations_state_space is project_innovations_state_space
