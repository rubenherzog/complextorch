import complextorch


def test_faes_common_primitives_are_exposed_at_package_root():
    """Validated reusable primitives are part of the root public API."""
    expected = {
        "downsample_innovations_state_space",
        "gaussian_instantaneous_information_rate",
        "gaussian_mutual_information_rate",
        "gaussian_transfer_entropy_rate",
        "innovations_spectral_density",
        "integrate_spectral_rate",
        "spectral_gaussian_mutual_information_rate",
        "spectral_gaussian_transfer_entropy_rate",
        "varma_to_innovations_state_space",
    }
    assert expected <= set(complextorch.__all__)
    for name in expected:
        function = getattr(complextorch, name)
        assert callable(function)
        assert function.__doc__


def test_pid_lattice_remains_private_measure_infrastructure():
    """PID lattice machinery is not promoted before a public PID measure exists."""
    assert "pid_lattice" not in complextorch.__all__
    assert not hasattr(complextorch, "pid_lattice")
