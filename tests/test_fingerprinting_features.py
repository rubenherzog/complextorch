import itertools

import torch

import complextorch
from complextorch import (
    cmem1_full_past,
    marginal_entropy_rate,
    maximum_temporal_mvgc,
    pairwise_temporal_mvgc,
    partial_information_rate_decomposition,
    pird_extrema,
    spectral_o_information_rate,
    synthetic_var,
    temporal_mvgc,
    var_to_innovations_state_space,
)


def _model(n=4, *, batch=False):
    rho = torch.tensor([0.45, 0.72], dtype=torch.float64) if batch else 0.62
    noise = torch.tensor([0.0, 0.2], dtype=torch.float64) if batch else 0.1
    return synthetic_var(
        "directed_ring",
        n,
        spectral_radius_target=rho,
        noise_correlation=noise,
        dtype=torch.float64,
    )


def _grid(n=257):
    return torch.linspace(0.0, 0.5, n, dtype=torch.float64)


def test_fingerprinting_feature_api_is_exported():
    names = (
        "cmem1_full_past",
        "pairwise_temporal_mvgc",
        "maximum_temporal_mvgc",
        "pird_extrema",
        "PIRDExtremaResult",
        "pairwise_gaussian_mutual_information_rate",
        "mean_pairwise_gaussian_mutual_information_rate",
    )
    for name in names:
        assert name in complextorch.__all__
        assert getattr(complextorch, name) is not None


def test_spectral_marginal_entropy_rate_matches_exact_dare_reductions():
    model = _model(batch=True)
    frequency = _grid(513)
    exact = marginal_entropy_rate(model, base=2.0, method="dare")
    spectral = marginal_entropy_rate(
        model,
        base=2.0,
        method="spectral",
        frequencies=frequency,
    )
    torch.testing.assert_close(spectral, exact, rtol=3e-6, atol=3e-6)


def test_cmem1_full_past_spectral_matches_exact_dare():
    model = _model(batch=True)
    frequency = _grid(513)
    exact = cmem1_full_past(model, base=2.0, marginal_method="dare")
    spectral = cmem1_full_past(
        model,
        base=2.0,
        marginal_method="spectral",
        frequencies=frequency,
    )
    torch.testing.assert_close(spectral, exact, rtol=3e-6, atol=3e-6)


def test_spectral_oir_spectrum_marginalization_matches_dare_path():
    system = var_to_innovations_state_space(_model())
    frequency = _grid(129)
    groups = ([0], [1, 2], [3])
    dare = spectral_o_information_rate(
        system, frequency, groups, base=2.0, marginalization="dare"
    )
    direct = spectral_o_information_rate(
        system, frequency, groups, base=2.0, marginalization="spectrum"
    )
    torch.testing.assert_close(direct, dare, rtol=2e-8, atol=2e-9)


def test_pairwise_temporal_mvgc_reuses_source_reductions_without_changing_values():
    model = _model(batch=True)
    matrix = pairwise_temporal_mvgc(model, base=2.0)
    expected = torch.zeros_like(matrix)
    for source in range(4):
        for target in range(4):
            if source == target:
                continue
            expected[..., source, target] = temporal_mvgc(
                model, source=(source,), target=(target,), base=2.0
            )
    torch.testing.assert_close(matrix, expected, rtol=2e-10, atol=2e-11)
    torch.testing.assert_close(
        maximum_temporal_mvgc(model, base=2.0),
        expected.amax(dim=(-2, -1)),
    )


def test_pird_extrema_matches_bruteforce_public_pird():
    system = var_to_innovations_state_space(_model())
    frequency = _grid(129)
    efficient = pird_extrema(system, frequency, base=2.0)

    synergy = []
    redundancy = []
    for target in range(4):
        for source0, source1 in itertools.combinations(
            [index for index in range(4) if index != target], 2
        ):
            result = partial_information_rate_decomposition(
                system,
                ([source0], [source1]),
                [target],
                frequency,
                base=2.0,
            )
            synergy.append(result.synergistic)
            redundancy.append(result.redundant)

    synergy = torch.stack(synergy, dim=-1)
    redundancy = torch.stack(redundancy, dim=-1)
    expected_syn, _ = synergy.max(dim=-1)
    expected_red, _ = redundancy.max(dim=-1)
    torch.testing.assert_close(efficient.synergistic, expected_syn, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(efficient.redundant, expected_red, rtol=0.0, atol=1e-12)
    # Multiple triples can tie by symmetry. Validate that the reported argmax
    # triples actually attain the corresponding maximum instead of requiring
    # one arbitrary tie-breaking order.
    syn_combo = tuple(int(x) for x in efficient.synergistic_indices.reshape(-1, 3)[0])
    red_combo = tuple(int(x) for x in efficient.redundant_indices.reshape(-1, 3)[0])
    syn_result = partial_information_rate_decomposition(
        system, ([syn_combo[0]], [syn_combo[1]]), [syn_combo[2]], frequency, base=2.0
    )
    red_result = partial_information_rate_decomposition(
        system, ([red_combo[0]], [red_combo[1]]), [red_combo[2]], frequency, base=2.0
    )
    torch.testing.assert_close(syn_result.synergistic, expected_syn, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(red_result.redundant, expected_red, rtol=0.0, atol=1e-12)


def test_pairwise_mir_spectral_matches_dare():
    from complextorch import (
        mean_pairwise_gaussian_mutual_information_rate,
        pairwise_gaussian_mutual_information_rate,
    )

    system = var_to_innovations_state_space(_model())
    frequencies = torch.linspace(0.0, 0.5, 1025, dtype=torch.float64)
    exact = pairwise_gaussian_mutual_information_rate(system, method="dare", base=2.0)
    spectral = pairwise_gaussian_mutual_information_rate(
        system, method="spectral", frequencies=frequencies, base=2.0
    )
    torch.testing.assert_close(spectral, exact, atol=2e-5, rtol=2e-5)
    indices = torch.triu_indices(exact.shape[-1], exact.shape[-1], 1)
    expected_mean = exact[..., indices[0], indices[1]].mean(-1)
    actual_mean = mean_pairwise_gaussian_mutual_information_rate(
        system, method="spectral", frequencies=frequencies, base=2.0
    )
    torch.testing.assert_close(actual_mean, expected_mean, atol=2e-5, rtol=2e-5)


def test_feature_helpers_preserve_float32_dtype():
    from complextorch import pairwise_gaussian_mutual_information_rate

    model = synthetic_var(
        "directed_ring",
        4,
        spectral_radius_target=0.55,
        noise_correlation=0.1,
        dtype=torch.float32,
    )
    system = var_to_innovations_state_space(model)
    frequency = torch.linspace(0.0, 0.5, 129, dtype=torch.float32)
    mir = pairwise_gaussian_mutual_information_rate(
        system, method="spectral", frequencies=frequency, base=2.0
    )
    gc = pairwise_temporal_mvgc(system, base=2.0)
    pird = pird_extrema(system, frequency, base=2.0)
    assert mir.dtype == torch.float32
    assert gc.dtype == torch.float32
    assert pird.synergistic.dtype == torch.float32
    assert torch.isfinite(mir).all()
    assert torch.isfinite(gc).all()
    assert torch.isfinite(pird.synergistic).all()
