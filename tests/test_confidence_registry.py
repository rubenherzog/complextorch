import torch

import complextorch
from complextorch.confidence import measure_confidence_intervals
from complextorch.inference_registry import InferenceMeasureConfig, evaluate_resampling_measures
from complextorch.measures.pdgc import spectral_partial_granger_causality_decomposition
from complextorch.measures.pird import spectral_partial_information_rate_decomposition
from complextorch.measures.primary import ModelMeasureConfig
from complextorch.representations import build_var_system
from complextorch.simulate import simulate_var
from complextorch.spectra import integrate_spectral_rate
from complextorch.var import VAR


def _system(dtype=torch.float64):
    coefficients = torch.tensor(
        [[[0.35, 0.00, 0.00], [0.08, 0.30, 0.00], [0.24, -0.15, 0.25]]],
        dtype=dtype,
    )
    covariance = torch.tensor(
        [[1.0, 0.10, 0.03], [0.10, 0.9, -0.04], [0.03, -0.04, 0.8]],
        dtype=dtype,
    )
    return build_var_system(coefficients, covariance)


def _config(dtype=torch.float64, *, half_open=False):
    frequencies = torch.linspace(0.0, 0.5, 65, dtype=dtype)
    primary = ModelMeasureConfig(
        frequencies=frequencies,
        source=(0,),
        target=(2,),
        base=torch.e,
    )
    return InferenceMeasureConfig(
        primary=primary,
        delta_oir_target_group=0,
        hop_sources=((0,), (1,)),
        hop_target=(2,),
        half_open=half_open,
    )


def test_registry_matches_direct_pird_pdgc_and_reuses_hop_tensors():
    system = _system()
    config = _config()
    tree = evaluate_resampling_measures(system, config)
    innovations = complextorch.var_to_innovations_state_space(system)
    frequencies = config.primary.frequencies

    direct_pird = spectral_partial_information_rate_decomposition(
        innovations, ((0,), (1,)), (2,), frequencies, base=torch.e
    )
    direct_pdgc = spectral_partial_granger_causality_decomposition(
        innovations, ((0,), (1,)), (2,), frequencies, base=torch.e
    )
    torch.testing.assert_close(tree["spectral_pird"]["atoms"], direct_pird.atoms)
    torch.testing.assert_close(tree["spectral_pdgc"]["atoms"], direct_pdgc.atoms)
    torch.testing.assert_close(
        tree["pird"]["atoms"],
        integrate_spectral_rate(direct_pird.atoms, frequencies),
    )
    torch.testing.assert_close(
        tree["pdgc"]["atoms"],
        integrate_spectral_rate(direct_pdgc.atoms, frequencies),
    )
    assert tree["hop"]["pird"]["atoms"] is tree["pird"]["atoms"]
    assert tree["hop"]["pdgc"]["atoms"] is tree["pdgc"]["atoms"]
    assert tree["spectral_hop"]["pird"]["atoms"] is tree["spectral_pird"]["atoms"]
    assert tree["spectral_hop"]["pdgc"]["atoms"] is tree["spectral_pdgc"]["atoms"]


def test_half_open_registry_integrates_same_spectral_atoms_with_faes_convention():
    system = _system()
    config = _config(half_open=True)
    tree = evaluate_resampling_measures(system, config)
    frequencies = config.primary.frequencies
    torch.testing.assert_close(
        tree["pird"]["atoms"],
        integrate_spectral_rate(
            tree["spectral_pird"]["atoms"], frequencies, half_open=True
        ),
    )
    torch.testing.assert_close(
        tree["pdgc"]["atoms"],
        integrate_spectral_rate(
            tree["spectral_pdgc"]["atoms"], frequencies, half_open=True
        ),
    )


def test_confidence_api_covers_rates_and_high_order_from_one_resampling_ensemble():
    system = _system()
    data = simulate_var(
        system.coefficients,
        system.innovation_covariance,
        350,
        burnin=200,
        seed=7,
    )[0]
    result = measure_confidence_intervals(
        data,
        measures=(
            "rates.o_information_rate",
            "rates.mutual_information_rate",
            "rates.transfer_entropy_rate",
            "pird.redundant",
            "pdgc.synergistic",
            "spectral_pird.delta",
            "spectral_pdgc.delta",
        ),
        var=VAR(order=1, mode="pooled", dtype="float64"),
        config=_config(),
        method="parametric",
        n_resamples=8,
        confidence=0.90,
        seed=11,
        return_samples=True,
    )
    assert result.n_valid >= 2
    assert result.n_valid + result.n_failed == 8
    for interval in result.intervals.values():
        assert interval.samples is not None
        assert interval.samples.shape[0] == result.n_valid
        assert torch.all(interval.lower <= interval.upper)


def test_confidence_api_all_compatible_does_not_require_oir_for_univariate_var():
    coefficients = torch.tensor([[[0.4]]], dtype=torch.float64)
    covariance = torch.tensor([[1.0]], dtype=torch.float64)
    data = simulate_var(coefficients, covariance, 250, burnin=100, seed=5)[0]
    result = measure_confidence_intervals(
        data,
        measures="all_compatible",
        var=VAR(order=1, mode="pooled", dtype="float64"),
        n_resamples=6,
        seed=3,
    )
    assert result.n_valid >= 2
    assert "rates.o_information_rate" not in result.intervals


def test_registry_preserves_leading_var_batch_axis():
    system = _system()
    batched = build_var_system(
        system.coefficients.expand(3, -1, -1, -1).clone(),
        system.innovation_covariance.expand(3, -1, -1).clone(),
    )
    tree = evaluate_resampling_measures(batched, _config())
    assert tree["rates"]["o_information_rate"].shape == (3,)
    assert tree["pird"]["atoms"].shape[0] == 3
    assert tree["pdgc"]["atoms"].shape[0] == 3
    assert tree["spectral_pird"]["atoms"].shape[0] == 3
    assert tree["spectral_pdgc"]["atoms"].shape[0] == 3
