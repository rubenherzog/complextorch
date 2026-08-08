import torch

from complextorch import (
    ModelMeasureConfig,
    compute_all_model_measures,
    delta_o_information_rate,
    gaussian_mutual_information_rate,
    o_information_rate,
    phiid_redundancy_from_model,
    spectral_hop_analysis,
    synthetic_var,
    temporal_mvgc,
    var_to_innovations_state_space,
)


def test_compute_all_model_measures_includes_rate_and_pairwise_families():
    rho = torch.tensor([0.4, 0.75], dtype=torch.float64)
    model = synthetic_var(
        "directed_ring",
        3,
        spectral_radius_target=rho,
        noise_correlation=torch.tensor([0.0, 0.25], dtype=torch.float64),
    )
    frequencies = torch.linspace(0.0, 0.5, 9, dtype=torch.float64)
    _, eigenvectors = torch.linalg.eigh(model.present_covariance)
    projection = eigenvectors[..., :, -1].unsqueeze(-2)

    result = compute_all_model_measures(
        model,
        ModelMeasureConfig(
            frequencies=frequencies,
            macro_projection=projection,
            base=2.0,
        ),
    )

    assert {"primitives", "rates", "mvgc"}.issubset(result["available"])
    assert result["primitives"]["innovations_state_space"] is result["context"].innovations
    assert result["primitives"]["observation_covariance"] is result["context"].observation_covariance
    assert result["primitives"]["autocovariances"] is result["context"].autocovariances
    assert result["primitives"]["cross_spectral_density"] is result["context"].cross_spectral_density

    rates = result["rates"]
    mvgc = result["mvgc"]
    assert rates["pairwise_mutual_information"].shape == (2, 3, 3)
    assert rates["mean_pairwise_mutual_information"].shape == (2,)
    assert rates["pairwise_transfer_entropy"].shape == (2, 3, 3)
    assert rates["pairwise_instantaneous_information"].shape == (2, 3, 3)
    assert rates["o_information"].shape == (2,)
    assert rates["delta_o_information"].shape == (2, 3)
    assert rates["pairwise_spectral_mutual_information"].shape == (2, 3, 3, 9)
    assert rates["pairwise_spectral_transfer_entropy"].shape == (2, 3, 3, 9)
    assert rates["spectral_o_information"].shape == (2, 9)
    assert rates["spectral_delta_o_information"].shape == (2, 3, 9)
    assert mvgc["pairwise_temporal"].shape == (2, 3, 3)
    assert mvgc["pairwise_spectral"].shape == (2, 3, 3, 9)

    transfer = result["primitives"]["transfer_function"]
    inverse_transfer = result["primitives"]["inverse_transfer_function"]
    assert transfer.shape == (2, 9, 3, 3)
    assert inverse_transfer.shape == transfer.shape
    identity = torch.eye(3, dtype=transfer.dtype).expand_as(transfer)
    torch.testing.assert_close(transfer @ inverse_transfer, identity, rtol=1e-8, atol=1e-8)

    for value in (
        rates["pairwise_mutual_information"],
        rates["mean_pairwise_mutual_information"],
        rates["pairwise_transfer_entropy"],
        rates["pairwise_instantaneous_information"],
        rates["o_information"],
        rates["delta_o_information"],
        rates["pairwise_spectral_mutual_information"],
        rates["pairwise_spectral_transfer_entropy"],
        rates["spectral_o_information"],
        rates["spectral_delta_o_information"],
        mvgc["pairwise_temporal"],
        mvgc["pairwise_spectral"],
    ):
        assert bool(torch.isfinite(value).all())

    innovations = var_to_innovations_state_space(model)
    torch.testing.assert_close(
        rates["pairwise_mutual_information"][..., 0, 1],
        gaussian_mutual_information_rate(innovations, (0,), (1,), base=2.0),
    )
    torch.testing.assert_close(
        rates["o_information"],
        o_information_rate(innovations, groups=None, base=2.0),
    )
    torch.testing.assert_close(
        rates["delta_o_information"][..., 0],
        delta_o_information_rate(innovations, target_group=0, groups=None, base=2.0),
    )
    torch.testing.assert_close(
        mvgc["pairwise_temporal"][..., 0, 1],
        temporal_mvgc(model, source=(0,), target=(1,), base=2.0),
    )
    torch.testing.assert_close(
        rates["pairwise_transfer_entropy"],
        0.5 * mvgc["pairwise_temporal"],
    )


def test_compute_all_mean_pairwise_mir_matches_unique_pair_mean():
    model = synthetic_var(
        "frustrated_ring",
        4,
        spectral_radius_target=0.65,
        noise_correlation=0.1,
        dtype=torch.float64,
    )
    result = compute_all_model_measures(model, ModelMeasureConfig(base=2.0))
    matrix = result["rates"]["pairwise_mutual_information"]
    indices = torch.triu_indices(4, 4, offset=1)
    expected = matrix[..., indices[0], indices[1]].mean(dim=-1)
    torch.testing.assert_close(
        result["rates"]["mean_pairwise_mutual_information"], expected
    )


def test_compute_all_phiid_reuses_one_cached_covariance_for_all_redundancies():
    model = synthetic_var(
        "directed_ring",
        3,
        spectral_radius_target=0.6,
        noise_correlation=0.15,
        dtype=torch.float64,
    )
    result = compute_all_model_measures(
        model,
        ModelMeasureConfig(
            phiid_variables=(0, 1),
            phiid_redundancies=("mmi", "ccs", "idep_a", "idep_b"),
            phiid_ccs_qmc_samples=64,
            base=2.0,
        ),
    )

    assert set(result["phiid_redundancy"]) == {"mmi", "ccs", "idep_a", "idep_b"}
    assert result["phiid_redundancy"]["mmi"] is result["phiid"]
    for backend, atoms in result["phiid_redundancy"].items():
        assert len(atoms) == 18
        assert bool(torch.isfinite(torch.stack(tuple(atoms.values()), dim=-1)).all())
        direct = phiid_redundancy_from_model(
            model,
            (0, 1),
            redundancy=backend,
            ccs_qmc_samples=64,
            base=2.0,
            autocovariance_sequence=result["context"].autocovariances,
        )
        torch.testing.assert_close(atoms["total"], direct["total"])
        torch.testing.assert_close(atoms["reconstruction"], direct["reconstruction"])


def test_compute_all_hop_exposes_pird_and_pdgc_without_second_dispatch():
    model = synthetic_var(
        "directed_ring",
        3,
        spectral_radius_target=0.55,
        noise_correlation=0.1,
        dtype=torch.float64,
    )
    frequencies = torch.linspace(0.0, 0.5, 9, dtype=torch.float64)
    config = ModelMeasureConfig(
        frequencies=frequencies,
        hop_sources=((0,), (1,)),
        hop_target=(2,),
        base=2.0,
    )
    result = compute_all_model_measures(model, config)

    assert {"hop", "pird", "pdgc"}.issubset(result["available"])
    assert result["hop"].pird is result["pird"]
    assert result["hop"].pdgc is result["pdgc"]
    assert result["spectral_hop"].pird is result["spectral_pird"]
    assert result["spectral_hop"].pdgc is result["spectral_pdgc"]

    direct = spectral_hop_analysis(
        var_to_innovations_state_space(model),
        ((0,), (1,)),
        (2,),
        frequencies,
        base=2.0,
    )
    torch.testing.assert_close(
        result["spectral_pird"].atoms,
        direct.pird.atoms,
    )
    torch.testing.assert_close(
        result["spectral_pdgc"].atoms,
        direct.pdgc.atoms,
    )


def test_compute_all_stochastic_interaction_accepts_var_via_canonical_conversion():
    model = synthetic_var(
        "directed_ring",
        3,
        spectral_radius_target=0.5,
        dtype=torch.float64,
    )
    result = compute_all_model_measures(
        model,
        ModelMeasureConfig(partition=((0,), (1, 2)), base=2.0),
    )
    assert "control" in result
    assert "stochastic_interaction" in result["control"]
    assert bool(torch.isfinite(result["control"]["stochastic_interaction"]).all())
