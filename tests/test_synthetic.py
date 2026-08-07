import pytest
import torch

from complextorch import ModelMeasureConfig, compute_all_model_measures
from complextorch.synthetic import (
    SYNTHETIC_SYSTEMS,
    available_synthetic_systems,
    equicorrelated_innovation_covariance,
    planted_module_projection,
    synthetic_system_parameters,
    synthetic_transition_matrix,
    synthetic_var,
)


@pytest.mark.parametrize("n_variables", [2, 3, 7])
@pytest.mark.parametrize("system", SYNTHETIC_SYSTEMS)
def test_all_synthetic_systems_support_arbitrary_n(system, n_variables):
    kwargs = {}
    if system in {"modular", "planted_modular"}:
        kwargs["n_modules"] = min(2, n_variables)
    transition = synthetic_transition_matrix(
        system, n_variables, dtype=torch.float64, **kwargs
    )
    assert transition.shape == (n_variables, n_variables)
    radius = torch.max(torch.abs(torch.linalg.eigvals(transition))).real
    torch.testing.assert_close(radius, torch.tensor(1.0, dtype=torch.float64))


@pytest.mark.parametrize(
    "system, expected",
    [
        ("directed_chain", {(1, 0), (2, 1), (3, 2)}),
        ("directed_ring", {(1, 0), (2, 1), (3, 2), (0, 3)}),
        ("hub_broadcast", {(1, 0), (2, 0), (3, 0)}),
        ("hub_convergent", {(0, 1), (0, 2), (0, 3)}),
    ],
)
def test_directed_topology_orientation_is_target_source(system, expected):
    transition = synthetic_transition_matrix(
        system,
        4,
        self_coupling=1.0,
        coupling=1.0,
        dtype=torch.float64,
    )
    off_diagonal = transition.clone()
    off_diagonal.fill_diagonal_(0.0)
    observed = {
        tuple(index.tolist())
        for index in torch.nonzero(off_diagonal, as_tuple=False)
    }
    assert observed == expected


def test_frustrated_ring_preserves_ring_graph_and_flips_one_sign():
    positive = synthetic_transition_matrix(
        "directed_ring", 5, self_coupling=0.0, coupling=1.0
    )
    frustrated = synthetic_transition_matrix(
        "frustrated_ring", 5, self_coupling=0.0, coupling=1.0
    )
    assert torch.equal(positive != 0, frustrated != 0)
    sign_difference = torch.sign(positive) != torch.sign(frustrated)
    assert int(sign_difference.sum()) == 1
    assert frustrated[1, 0] < 0


def test_planted_modular_is_block_diagonal_and_projection_is_orthonormal():
    transition = synthetic_transition_matrix(
        "planted_modular",
        7,
        n_modules=3,
        self_coupling=1.0,
        within_coupling=0.4,
    )
    labels = torch.div(torch.arange(7) * 3, 7, rounding_mode="floor")
    between = labels[:, None] != labels[None, :]
    assert torch.count_nonzero(transition[between]) == 0

    projection = planted_module_projection(7, 3)
    torch.testing.assert_close(
        projection @ projection.T,
        torch.eye(3, dtype=projection.dtype),
    )


def test_equicorrelated_covariance_has_exact_requested_correlation():
    q = equicorrelated_innovation_covariance(
        5,
        torch.tensor([-0.2, 0.0, 0.7], dtype=torch.float64),
        variance=2.5,
    )
    assert q.shape == (3, 5, 5)
    torch.testing.assert_close(
        torch.diagonal(q, dim1=-2, dim2=-1),
        torch.full((3, 5), 2.5, dtype=torch.float64),
    )
    torch.testing.assert_close(q[0, 0, 1], torch.tensor(-0.5, dtype=torch.float64))
    torch.testing.assert_close(q[1, 0, 1], torch.tensor(0.0, dtype=torch.float64))
    torch.testing.assert_close(q[2, 0, 1], torch.tensor(1.75, dtype=torch.float64))
    assert bool((torch.linalg.eigvalsh(q) > 0).all())


def test_equicorrelation_rejects_positive_definite_boundary():
    with pytest.raises(ValueError, match="correlation must satisfy"):
        equicorrelated_innovation_covariance(4, -1.0 / 3.0)
    with pytest.raises(ValueError, match="correlation must satisfy"):
        equicorrelated_innovation_covariance(4, 1.0)


def test_synthetic_var_broadcasts_parameter_grid_into_batch():
    rho = torch.linspace(0.2, 0.9, 4, dtype=torch.float64)[:, None]
    correlation = torch.linspace(-0.1, 0.6, 3, dtype=torch.float64)[None, :]
    model = synthetic_var(
        "directed_ring",
        5,
        spectral_radius_target=rho,
        noise_correlation=correlation,
    )
    assert model.batch_size == 12
    assert model.order == 1
    assert model.n_variables == 5
    expected_rho = rho.expand(4, 3).reshape(-1)
    torch.testing.assert_close(model.spectral_radius, expected_rho)
    assert model.coefficients.shape == (12, 1, 5, 5)
    assert model.innovation_covariance.shape == (12, 5, 5)


def test_parameter_grid_runs_all_primary_measures_with_pc1_macro():
    """Exercise the intended grid -> model -> primary-measure workflow in batch."""

    n_variables = 4
    rho = torch.tensor([0.35, 0.8], dtype=torch.float64)[:, None]
    correlation = torch.tensor([-0.15, 0.0, 0.45], dtype=torch.float64)[None, :]
    model = synthetic_var(
        "directed_ring",
        n_variables,
        spectral_radius_target=rho,
        noise_correlation=correlation,
    )
    batch = rho.numel() * correlation.numel()
    assert model.batch_size == batch

    # Deterministic one-dimensional coarse-graining: the leading stationary PC.
    _, eigenvectors = torch.linalg.eigh(model.present_covariance)
    pc1 = eigenvectors[..., -1].unsqueeze(-2)
    assert pc1.shape == (batch, 1, n_variables)
    torch.testing.assert_close(
        pc1 @ pc1.transpose(-1, -2),
        torch.ones((batch, 1, 1), dtype=pc1.dtype),
    )

    frequencies = torch.linspace(0.0, 0.5, 5, dtype=torch.float64)
    config = ModelMeasureConfig(
        frequencies=frequencies,
        autocovariance_max_lag=1,
        ais_lag=1,
        cmem_max_lag=1,
        cmem_decomposition_max_lag=1,
        source=(0,),
        target=(1,),
        phiid_variables=(0, 1),
        phiid_lag=1,
        macro_projection=pc1,
        base=2.0,
    )
    result = compute_all_model_measures(model, config)

    expected_available = {
        "entropy_rate",
        "criticality",
        "cross_spectral_density",
        "spectral_entropy",
        "gaussian",
        "autocovariances",
        "predictive_information",
        "active_information_storage",
        "cmem",
        "phiid",
        "emergence",
        "mvgc",
        "control",
    }
    assert expected_available.issubset(set(result["available"]))
    assert "dynamical_dependence" not in result["not_available"]

    # The quantities most directly used for phase diagrams must retain the
    # synthetic batch axis and be finite for the entire grid.
    scalar_batch_quantities = {
        "entropy_rate": result["dynamics"]["entropy_rate"],
        "spectral_radius": result["criticality"]["spectral_radius"],
        "stability_margin": result["criticality"]["stability_margin"],
        "dominant_timescale": result["criticality"]["dominant_timescale"],
        "covariance_amplification": result["criticality"]["covariance_amplification"],
        "entropy": result["gaussian"]["entropy"],
        "total_correlation": result["gaussian"]["total_correlation"],
        "dual_total_correlation": result["gaussian"]["dual_total_correlation"],
        "o_information": result["gaussian"]["o_information"],
        "s_information": result["gaussian"]["s_information"],
        "predictive_information": result["dynamics"]["predictive_information"],
        "dynamical_dependence": result["control"]["dynamical_dependence"],
        "temporal_mvgc": result["mvgc"]["temporal"],
    }
    for name, value in scalar_batch_quantities.items():
        assert value.shape == (batch,), name
        assert bool(torch.isfinite(value).all()), name

    per_variable_quantities = {
        "active_information_storage": result["dynamics"]["active_information_storage"],
        "spectral_entropy": result["frequency"]["spectral_entropy"],
    }
    for name, value in per_variable_quantities.items():
        assert value.shape == (batch, n_variables), name
        assert bool(torch.isfinite(value).all()), name

    assert result["gaussian"]["pairwise_mutual_information"].shape == (
        batch,
        n_variables,
        n_variables,
    )
    assert result["autocovariances"].shape == (batch, 2, n_variables, n_variables)
    assert result["frequency"]["cross_spectral_density"].shape[:2] == (
        batch,
        frequencies.numel(),
    )
    assert result["mvgc"]["spectral"].shape == (batch, frequencies.numel())

    # A flattened batch can be mapped losslessly back onto the original grid.
    o_information_grid = result["gaussian"]["o_information"].reshape(
        rho.shape[0], correlation.shape[1]
    )
    dd_grid = result["control"]["dynamical_dependence"].reshape(
        rho.shape[0], correlation.shape[1]
    )
    assert o_information_grid.shape == (2, 3)
    assert dd_grid.shape == (2, 3)

    # Verify batch semantics numerically at one grid point against the same
    # system constructed independently.
    i_rho, i_corr = 1, 2
    flat_index = i_rho * correlation.shape[1] + i_corr
    single = synthetic_var(
        "directed_ring",
        n_variables,
        spectral_radius_target=rho[i_rho, 0],
        noise_correlation=correlation[0, i_corr],
    )
    _, single_eigenvectors = torch.linalg.eigh(single.present_covariance)
    single_pc1 = single_eigenvectors[..., -1].unsqueeze(-2)
    single_result = compute_all_model_measures(
        single,
        ModelMeasureConfig(
            autocovariance_max_lag=1,
            ais_lag=1,
            cmem_max_lag=1,
            cmem_decomposition_max_lag=1,
            source=(0,),
            target=(1,),
            phiid_variables=(0, 1),
            phiid_lag=1,
            macro_projection=single_pc1,
            base=2.0,
        ),
    )
    torch.testing.assert_close(
        result["gaussian"]["o_information"][flat_index],
        single_result["gaussian"]["o_information"].squeeze(0),
    )
    torch.testing.assert_close(
        result["control"]["dynamical_dependence"][flat_index],
        single_result["control"]["dynamical_dependence"].squeeze(0),
    )


def test_nonnormal_coupling_changes_structure_without_changing_template_radius():
    coupling = torch.tensor([0.0, 0.5, 2.0], dtype=torch.float64)
    transition = synthetic_transition_matrix(
        "nonnormal_feedforward",
        5,
        self_coupling=1.0,
        coupling=coupling,
    )
    assert transition.shape == (3, 5, 5)
    radius = torch.max(torch.abs(torch.linalg.eigvals(transition)), dim=-1).values.real
    torch.testing.assert_close(radius, torch.ones(3, dtype=torch.float64))
    eye = torch.eye(5, dtype=torch.float64)
    assert torch.linalg.matrix_norm(transition[2] - eye) > torch.linalg.matrix_norm(
        transition[1] - eye
    )


def test_random_systems_are_seed_reproducible():
    for system in ("erdos_renyi", "random_directed"):
        first = synthetic_transition_matrix(system, 8, density=0.4, seed=17)
        second = synthetic_transition_matrix(system, 8, density=0.4, seed=17)
        different = synthetic_transition_matrix(system, 8, density=0.4, seed=18)
        torch.testing.assert_close(first, second)
        assert not torch.equal(first, different)


def test_aliases_and_parameter_introspection():
    torch.testing.assert_close(
        synthetic_transition_matrix("ring", 4),
        synthetic_transition_matrix("directed_ring", 4),
    )
    torch.testing.assert_close(
        synthetic_transition_matrix("random_network", 6, seed=3),
        synthetic_transition_matrix("random_directed", 6, seed=3),
    )
    assert available_synthetic_systems() == SYNTHETIC_SYSTEMS
    assert synthetic_system_parameters("ring") == ("self_coupling", "coupling")
    assert "density" in synthetic_system_parameters("random_network")


def test_dtype_is_preserved():
    model = synthetic_var(
        "fully_connected",
        4,
        spectral_radius_target=torch.tensor(0.7, dtype=torch.float32),
        noise_correlation=torch.tensor(0.2, dtype=torch.float32),
        dtype=torch.float32,
    )
    assert model.coefficients.dtype == torch.float32
    assert model.innovation_covariance.dtype == torch.float32


def test_invalid_topology_and_parameters_fail_explicitly():
    with pytest.raises(ValueError, match="unknown synthetic system"):
        synthetic_var("not_a_system", 4)
    with pytest.raises(ValueError, match="n_variables"):
        synthetic_var("ring", 1)
    with pytest.raises(ValueError, match="spectral_radius_target"):
        synthetic_var("ring", 4, spectral_radius_target=1.0)
    with pytest.raises(ValueError, match="density"):
        synthetic_transition_matrix("erdos_renyi", 4, density=1.2)
    with pytest.raises(ValueError, match="n_modules"):
        synthetic_transition_matrix("modular", 4, n_modules=5)
    with pytest.raises(ValueError, match="zero spectral radius"):
        synthetic_transition_matrix(
            "directed_chain", 4, self_coupling=0.0, coupling=1.0
        )
