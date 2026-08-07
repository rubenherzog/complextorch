"""Public-import regression tests for dynamical-dependence optimisation."""

import complextorch as ct


def test_dd_optimization_public_api_is_exported():
    """The parity optimiser primitives must be available from the package root."""
    names = (
        "DDGradientSearchResult",
        "innovations_proxy_sequence",
        "orthonormalise_projection",
        "proxy_dynamical_dependence",
        "proxy_dynamical_dependence_gradient",
        "spectral_dynamical_dependence",
        "spectral_dynamical_dependence_gradient",
        "optimise_dynamical_dependence_proxy",
        "optimise_dynamical_dependence_spectral",
    )
    for name in names:
        assert hasattr(ct, name)
        assert name in ct.__all__
