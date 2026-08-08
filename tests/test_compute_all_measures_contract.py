import torch

from complextorch import (
    ModelMeasureConfig,
    compute_all_model_measures,
    gaussian_mutual_information_rate,
    o_information_rate,
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
    assert rates["pairwise_spectral_mutual_information"].shape == (2, 3, 3, 9)
    assert rates["pairwise_spectral_transfer_entropy"].shape == (2, 3, 3, 9)
    assert rates["spectral_o_information"].shape == (2, 9)
    assert mvgc["pairwise_temporal"].shape == (2, 3, 3)
    assert mvgc["pairwise_spectral"].shape == (2, 3, 3, 9)

    for value in (
        rates["pairwise_mutual_information"],
        rates["mean_pairwise_mutual_information"],
        rates["pairwise_transfer_entropy"],
        rates["pairwise_instantaneous_information"],
        rates["o_information"],
        rates["pairwise_spectral_mutual_information"],
        rates["pairwise_spectral_transfer_entropy"],
        rates["spectral_o_information"],
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
