import complextorch


def test_root_dd_optimizer_api_is_unified_and_complexbox_legacy_is_preserved():
    assert hasattr(complextorch, "optimise_dynamical_dependence")
    assert hasattr(complextorch, "optimise_dynamical_dependence_proxy")
    assert hasattr(complextorch, "optimise_dynamical_dependence_spectral")

    # New optimizer backends remain implementation details. Users select them
    # through optimise_dynamical_dependence(..., optimizer=...).
    assert not hasattr(
        complextorch, "optimise_dynamical_dependence_proxy_riemannian"
    )
    assert not hasattr(
        complextorch, "optimise_dynamical_dependence_spectral_riemannian"
    )
    assert not hasattr(complextorch, "DDRiemannianSearchResult")
