import torch

from complextorch import (
    InnovationsStateSpace,
    ModelMeasureConfig,
    build_measure_context,
    compute_all_model_measures,
    integrate_spectral_rate,
    marginal_entropy_rate,
    reduce_innovations_state_space,
    spectral_entropy_rate,
    spectral_entropy_rate_from_spectrum,
    synthetic_var,
    var_to_innovations_state_space,
)
from complextorch.measures.gaussian import gaussian_entropy
from complextorch.spectra import innovations_spectral_density


def test_marginal_entropy_rate_returns_exact_per_variable_vector_for_batch():
    model = synthetic_var(
        "directed_ring",
        3,
        spectral_radius_target=torch.tensor([0.45, 0.7], dtype=torch.float64),
        noise_correlation=torch.tensor([0.0, 0.2], dtype=torch.float64),
    )
    innovations = var_to_innovations_state_space(model)

    values = marginal_entropy_rate(model, base=2.0)

    assert values.shape == (2, 3)
    assert bool(torch.isfinite(values).all())
    expected = torch.stack(
        [
            gaussian_entropy(
                reduce_innovations_state_space(
                    innovations, (index,)
                ).innovation_covariance,
                base=2.0,
            )
            for index in range(3)
        ],
        dim=-1,
    )
    torch.testing.assert_close(values, expected)


def test_entropy_rate_extensions_preserve_unbatched_shapes():
    batched = var_to_innovations_state_space(
        synthetic_var(
            "directed_ring",
            3,
            spectral_radius_target=0.55,
            dtype=torch.float64,
        )
    )
    system = InnovationsStateSpace(
        batched.transition[0],
        batched.observation[0],
        batched.gain[0],
        batched.innovation_covariance[0],
    )
    frequencies = torch.linspace(0.0, 0.5, 17, dtype=torch.float64)

    assert marginal_entropy_rate(system).shape == (3,)
    assert spectral_entropy_rate(system, frequencies).shape == (17,)


def test_spectral_entropy_rate_matches_spectrum_formula_and_integrates_to_broadband():
    sampling_frequency = 200.0
    model = synthetic_var(
        "frustrated_ring",
        3,
        spectral_radius_target=0.6,
        noise_correlation=0.15,
        dtype=torch.float64,
    )
    innovations = var_to_innovations_state_space(model)
    frequencies = torch.linspace(
        0.0,
        sampling_frequency / 2.0,
        4097,
        dtype=torch.float64,
    )

    spectral = spectral_entropy_rate(
        innovations,
        frequencies,
        sampling_frequency=sampling_frequency,
        base=2.0,
    )
    spectrum = innovations_spectral_density(
        innovations,
        frequencies,
        sampling_frequency=sampling_frequency,
    )
    direct = spectral_entropy_rate_from_spectrum(
        spectrum,
        sampling_frequency=sampling_frequency,
        base=2.0,
    )

    # ``synthetic_var`` preserves an explicit batch dimension, even for one
    # parameter setting, so frequency-resolved outputs retain that batch axis.
    assert spectral.shape == (1, 4097)
    torch.testing.assert_close(spectral, direct)

    integrated = integrate_spectral_rate(
        spectral,
        frequencies,
        sampling_frequency=sampling_frequency,
    )
    expected = gaussian_entropy(innovations.innovation_covariance, base=2.0)
    torch.testing.assert_close(integrated, expected, rtol=2e-6, atol=2e-6)


def test_measure_context_uses_physical_frequency_spectral_convention():
    sampling_frequency = 200.0
    model = synthetic_var(
        "directed_ring",
        3,
        spectral_radius_target=0.5,
        dtype=torch.float64,
    )
    frequencies = torch.linspace(0.0, 100.0, 21, dtype=torch.float64)
    context = build_measure_context(
        model,
        ModelMeasureConfig(
            frequencies=frequencies,
            sampling_frequency=sampling_frequency,
        ),
    )
    expected = innovations_spectral_density(
        var_to_innovations_state_space(model),
        frequencies,
        sampling_frequency=sampling_frequency,
    )
    torch.testing.assert_close(context.cross_spectral_density, expected)


def test_compute_all_model_measures_exposes_entropy_rate_extensions_without_band_outputs():
    model = synthetic_var(
        "directed_ring",
        4,
        spectral_radius_target=0.55,
        noise_correlation=0.1,
        dtype=torch.float64,
    )
    frequencies = torch.linspace(0.0, 0.5, 17, dtype=torch.float64)

    result = compute_all_model_measures(
        model,
        ModelMeasureConfig(frequencies=frequencies, base=2.0),
    )

    assert result["dynamics"]["marginal_entropy_rate"].shape == (1, 4)
    assert result["frequency"]["spectral_entropy_rate"].shape == (1, 17)
    assert "marginal_entropy_rate" in result["available"]
    assert "spectral_entropy_rate" in result["available"]
    assert not any("band" in key for key in result["frequency"])

    torch.testing.assert_close(
        result["dynamics"]["marginal_entropy_rate"],
        marginal_entropy_rate(model, base=2.0),
    )
    torch.testing.assert_close(
        result["frequency"]["spectral_entropy_rate"],
        spectral_entropy_rate(model, frequencies, base=2.0),
    )
