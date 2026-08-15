import torch

from complextorch import ModelMeasureConfig, build_var_system, compute_all_model_measures, demo_var
from complextorch.measures.primary import build_measure_context


def _config():
    return ModelMeasureConfig(
        frequencies=torch.linspace(0.0, 0.5, 257, dtype=torch.float64),
        autocovariance_max_lag=3,
        ais_lag=2,
        cmem_max_lag=3,
        cmem_decomposition_max_lag=2,
        phiid_variables=(0, 1),
        phiid_lag=3,
        source=(1,),
        target=(0,),
        conditional=(2,),
        macro_projection=torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]], dtype=torch.float64
        ),
    )


def test_equivalent_var_and_state_space_share_observable_measures():
    coefficients, covariance = demo_var(n_variables=3, order=2)
    var = build_var_system(coefficients, covariance)
    state_space = var.to_state_space()
    config = _config()

    var_result = compute_all_model_measures(var, config)
    state_result = compute_all_model_measures(state_space, config)

    for family, name in (
        ("gaussian", "covariance"),
        ("dynamics", "entropy_rate"),
        ("dynamics", "predictive_information"),
        ("dynamics", "active_information_storage"),
        ("frequency", "spectral_entropy"),
        ("mvgc", "temporal"),
        ("control", "dynamical_dependence"),
    ):
        torch.testing.assert_close(
            state_result[family][name], var_result[family][name], rtol=1e-7, atol=1e-9
        )

    torch.testing.assert_close(
        state_result["frequency"]["cross_spectral_density"],
        var_result["frequency"]["cross_spectral_density"],
        rtol=1e-7,
        atol=1e-9,
    )
    for name in var_result["phiid"]:
        torch.testing.assert_close(
            state_result["phiid"][name], var_result["phiid"][name], rtol=1e-7, atol=1e-9
        )
    for name in ("psi", "delta", "gamma"):
        torch.testing.assert_close(
            state_result["emergence"][name],
            var_result["emergence"][name],
            rtol=1e-7,
            atol=1e-9,
        )


def test_all_delay_families_share_one_autocovariance_sequence():
    coefficients, covariance = demo_var(n_variables=3, order=2)
    var = build_var_system(coefficients, covariance)
    context = build_measure_context(var, _config())
    assert context.max_lag == 3
    assert context.autocovariances.shape[1] == 4


def test_availability_metadata_is_explicit():
    coefficients, covariance = demo_var(n_variables=3, order=2)
    var = build_var_system(coefficients, covariance)
    result = compute_all_model_measures(var, _config())
    assert "cmem" in result["available"]
    assert "mvgc" in result["available"]
    assert "control" in result["available"]
    assert "dynamical_dependence" in result["control"]


def test_equivalent_var_and_innovations_state_space_share_covariance_backbone():
    from complextorch import var_to_innovations_state_space
    from complextorch.measures.primary import model_autocovariances

    coefficients, covariance = demo_var(n_variables=3, order=2)
    var = build_var_system(coefficients, covariance)
    innovations = var_to_innovations_state_space(var)

    expected = model_autocovariances(var, 3)
    actual = model_autocovariances(innovations, 3)
    torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-9)


def test_equivalent_var_and_innovations_state_space_share_observable_measures():
    from complextorch import var_to_innovations_state_space

    coefficients, covariance = demo_var(n_variables=3, order=2)
    var = build_var_system(coefficients, covariance)
    innovations = var_to_innovations_state_space(var)
    config = _config()

    var_result = compute_all_model_measures(var, config)
    innovations_result = compute_all_model_measures(innovations, config)

    for family, name in (
        ("gaussian", "covariance"),
        ("dynamics", "entropy_rate"),
        ("dynamics", "predictive_information"),
        ("dynamics", "active_information_storage"),
        ("criticality", "covariance_amplification"),
    ):
        torch.testing.assert_close(
            innovations_result[family][name],
            var_result[family][name],
            rtol=1e-7,
            atol=1e-9,
        )

    for name in innovations_result["phiid"]:
        torch.testing.assert_close(
            innovations_result["phiid"][name],
            var_result["phiid"][name],
            rtol=1e-7,
            atol=1e-9,
        )
    for name in ("psi", "delta", "gamma"):
        torch.testing.assert_close(
            innovations_result["emergence"][name],
            var_result["emergence"][name],
            rtol=1e-7,
            atol=1e-9,
        )
    assert not {
        "gaussian",
        "autocovariances",
        "predictive_information",
        "active_information_storage",
        "cmem",
        "phiid",
        "emergence",
    }.intersection(innovations_result["not_available"])


def test_innovations_autocovariances_preserve_batch_dtype_and_device():
    from complextorch import InnovationsStateSpace
    from complextorch.measures.primary import model_autocovariances

    dtype = torch.float32
    transition = torch.tensor(
        [[[0.2, 0.0], [0.1, 0.3]], [[0.5, 0.1], [0.0, 0.4]]], dtype=dtype
    )
    observation = torch.eye(2, dtype=dtype).expand(2, -1, -1)
    gain = torch.tensor(
        [[[0.8, 0.0], [0.1, 0.7]], [[0.6, 0.1], [0.0, 0.9]]], dtype=dtype
    )
    innovation_covariance = torch.tensor(
        [[[1.0, 0.2], [0.2, 0.8]], [[0.7, 0.1], [0.1, 1.1]]], dtype=dtype
    )
    system = InnovationsStateSpace(
        transition, observation, gain, innovation_covariance
    )
    gamma = model_autocovariances(system, 2)
    assert gamma.shape == (2, 3, 2, 2)
    assert gamma.dtype == dtype
    assert gamma.device == transition.device
    assert bool(torch.isfinite(gamma).all())
