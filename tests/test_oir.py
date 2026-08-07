import math

import pytest
import torch

import complextorch
from complextorch import (
    InnovationsStateSpace,
    build_var_system,
    delta_o_information_rate,
    innovations_spectral_density,
    integrate_spectral_rate,
    o_information_rate,
    spectral_delta_o_information_rate,
    spectral_o_information_rate,
    var_to_innovations_state_space,
)
from complextorch.measures.gaussian import o_information
from complextorch.spectra import hermitian_logdet


def _iss(coefficients, covariance, *, dtype=torch.float64):
    return var_to_innovations_state_space(
        build_var_system(
            torch.as_tensor(coefficients, dtype=dtype),
            torch.as_tensor(covariance, dtype=dtype),
        )
    )


def _dynamic_three(dtype=torch.float64):
    coefficients = torch.tensor(
        [
            [
                [0.42, 0.08, 0.00],
                [0.03, 0.35, -0.06],
                [0.24, 0.14, 0.28],
            ]
        ],
        dtype=dtype,
    )
    covariance = torch.tensor(
        [[1.0, 0.18, 0.05], [0.18, 0.9, -0.08], [0.05, -0.08, 0.8]],
        dtype=dtype,
    )
    return _iss(coefficients, covariance, dtype=dtype)


def _dynamic_four(dtype=torch.float64):
    coefficients = torch.tensor(
        [
            [
                [0.38, 0.04, 0.00, 0.00],
                [0.06, 0.32, -0.04, 0.00],
                [0.00, 0.10, 0.29, 0.03],
                [0.20, -0.08, 0.12, 0.27],
            ]
        ],
        dtype=dtype,
    )
    covariance = torch.tensor(
        [
            [1.0, 0.12, 0.02, 0.04],
            [0.12, 0.9, -0.06, 0.01],
            [0.02, -0.06, 0.8, 0.08],
            [0.04, 0.01, 0.08, 0.85],
        ],
        dtype=dtype,
    )
    return _iss(coefficients, covariance, dtype=dtype)


def _grid(dtype=torch.float64, n=513):
    return torch.linspace(0.0, 0.5, n, dtype=dtype)


def _scalar(value):
    return value.reshape(-1)[0]


def _spectral_oir_from_full_psd(spectrum, groups):
    """Independent OIR oracle formed only from full-spectrum submatrices."""
    groups = tuple(tuple(group) for group in groups)
    n_groups = len(groups)
    all_indices = tuple(index for group in groups for index in group)

    def sub_logdet(indices):
        """Return logdet of a PSD submatrix without state-space reduction."""
        index = torch.tensor(indices, device=spectrum.device)
        submatrix = spectrum.index_select(-2, index).index_select(-1, index)
        return hermitian_logdet(submatrix)

    singletons = torch.stack([sub_logdet(group) for group in groups], dim=-1).sum(-1)
    complements = torch.stack(
        [
            sub_logdet(
                tuple(
                    channel
                    for group_index, group in enumerate(groups)
                    if group_index != excluded
                    for channel in group
                )
            )
            for excluded in range(n_groups)
        ],
        dim=-1,
    ).sum(-1)
    return 0.5 * (
        singletons
        - complements
        + (n_groups - 2) * sub_logdet(all_indices)
    )


def test_public_oir_api_is_exported_and_documented():
    names = (
        "o_information_rate",
        "spectral_o_information_rate",
        "delta_o_information_rate",
        "spectral_delta_o_information_rate",
    )
    for name in names:
        assert name in complextorch.__all__
        function = getattr(complextorch, name)
        assert callable(function)
        assert function.__doc__


def test_iid_oir_equals_static_gaussian_o_information():
    covariance = torch.tensor(
        [[1.0, 0.35, 0.20], [0.35, 0.9, 0.25], [0.20, 0.25, 1.1]],
        dtype=torch.float64,
    )
    system = _iss(torch.zeros((1, 3, 3), dtype=torch.float64), covariance)
    dynamic = _scalar(o_information_rate(system, base=2.0))
    static = o_information(covariance, base=2.0)
    torch.testing.assert_close(dynamic, static, rtol=1e-10, atol=1e-11)


def test_redundant_and_synergistic_iid_systems_have_expected_signs():
    redundant_covariance = torch.full((3, 3), 0.55, dtype=torch.float64)
    redundant_covariance.fill_diagonal_(1.0)
    synergistic_covariance = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 2.5]],
        dtype=torch.float64,
    )
    zero = torch.zeros((1, 3, 3), dtype=torch.float64)
    redundant = _scalar(o_information_rate(_iss(zero, redundant_covariance)))
    synergistic = _scalar(o_information_rate(_iss(zero, synergistic_covariance)))
    assert redundant > 0.05
    assert synergistic < -0.1


def test_two_group_oir_is_zero_and_grouping_is_permutation_invariant():
    system = _dynamic_four()
    two_group = o_information_rate(system, groups=([0, 1], [2, 3]))
    torch.testing.assert_close(
        two_group, torch.zeros_like(two_group), rtol=0.0, atol=2e-9
    )
    reference = o_information_rate(system, groups=([0], [1], [2, 3]))
    permuted = o_information_rate(system, groups=([2, 3], [0], [1]))
    torch.testing.assert_close(permuted, reference, rtol=2e-9, atol=2e-10)


def test_spectral_oir_matches_full_psd_subblock_oracle():
    system = _dynamic_four()
    frequency = _grid(n=257)
    groups = ([0], [1, 2], [3])
    spectrum = innovations_spectral_density(system, frequency)
    direct = _spectral_oir_from_full_psd(spectrum, groups)
    reduced = spectral_o_information_rate(system, frequency, groups)
    torch.testing.assert_close(reduced, direct, rtol=2e-8, atol=2e-9)


def test_spectral_oir_integrates_to_temporal_at_nonunit_sampling_frequency():
    system = _dynamic_three()
    sampling_frequency = 4.0
    frequency = torch.linspace(
        0.0, sampling_frequency / 2.0, 1025, dtype=torch.float64
    )
    spectral = spectral_o_information_rate(
        system, frequency, sampling_frequency=sampling_frequency
    )
    temporal = o_information_rate(system)
    integrated = integrate_spectral_rate(
        spectral, frequency, sampling_frequency=sampling_frequency
    )
    torch.testing.assert_close(integrated, temporal, rtol=2e-7, atol=3e-9)


def test_delta_oir_equals_difference_of_nested_oirs():
    system = _dynamic_four()
    groups = ([0], [1], [2], [3])
    full = o_information_rate(system, groups)
    for target in range(4):
        reduced_groups = tuple(
            group for index, group in enumerate(groups) if index != target
        )
        expected = full - o_information_rate(system, reduced_groups)
        actual = delta_o_information_rate(system, target, groups)
        torch.testing.assert_close(actual, expected, rtol=3e-9, atol=3e-10)


def test_spectral_delta_matches_pointwise_nested_oir_difference_and_integral():
    system = _dynamic_four()
    frequency = _grid(n=513)
    groups = ([0], [1], [2], [3])
    target = 2
    reduced_groups = tuple(
        group for index, group in enumerate(groups) if index != target
    )
    expected_spectrum = spectral_o_information_rate(
        system, frequency, groups
    ) - spectral_o_information_rate(system, frequency, reduced_groups)
    actual_spectrum = spectral_delta_o_information_rate(
        system, frequency, target, groups
    )
    torch.testing.assert_close(
        actual_spectrum, expected_spectrum, rtol=3e-8, atol=3e-9
    )
    temporal = delta_o_information_rate(system, target, groups)
    torch.testing.assert_close(
        integrate_spectral_rate(actual_spectrum, frequency),
        temporal,
        rtol=8e-7,
        atol=8e-9,
    )


def test_group_subset_marginalises_unlisted_channels_exactly():
    system = _dynamic_four()
    grouped_from_full = o_information_rate(system, groups=([0], [1], [3]))
    from complextorch.control import reduce_innovations_state_space

    reduced = reduce_innovations_state_space(system, [0, 1, 3])
    directly_reduced = o_information_rate(reduced)
    torch.testing.assert_close(
        grouped_from_full, directly_reduced, rtol=2e-9, atol=2e-10
    )


def test_all_oir_functions_batch_match_explicit_system_loop():
    coefficients = torch.tensor(
        [
            [[0.40, 0.04, 0.00], [0.02, 0.32, 0.03], [0.18, 0.10, 0.25]],
            [[0.30, -0.05, 0.02], [0.06, 0.36, 0.00], [0.12, -0.08, 0.28]],
            [[0.35, 0.00, -0.03], [0.08, 0.27, 0.05], [0.16, 0.07, 0.31]],
        ],
        dtype=torch.float64,
    ).unsqueeze(1)
    covariance = torch.tensor(
        [
            [[1.0, 0.10, 0.02], [0.10, 0.9, -0.03], [0.02, -0.03, 0.8]],
            [[0.9, -0.06, 0.04], [-0.06, 1.1, 0.07], [0.04, 0.07, 0.85]],
            [[1.2, 0.05, -0.02], [0.05, 0.8, 0.04], [-0.02, 0.04, 0.95]],
        ],
        dtype=torch.float64,
    )
    batched = _iss(coefficients, covariance)
    frequency = _grid(n=129)
    batch_temporal = o_information_rate(batched)
    batch_spectral = spectral_o_information_rate(batched, frequency)
    batch_delta = delta_o_information_rate(batched, 1)
    batch_spectral_delta = spectral_delta_o_information_rate(
        batched, frequency, 1
    )

    assert batch_temporal.shape == (3,)
    assert batch_spectral.shape == (3, frequency.numel())
    assert batch_delta.shape == (3,)
    assert batch_spectral_delta.shape == (3, frequency.numel())
    for index in range(3):
        single = InnovationsStateSpace(
            batched.transition[index],
            batched.observation[index],
            batched.gain[index],
            batched.innovation_covariance[index],
        )
        torch.testing.assert_close(batch_temporal[index], o_information_rate(single))
        torch.testing.assert_close(
            batch_spectral[index], spectral_o_information_rate(single, frequency)
        )
        torch.testing.assert_close(
            batch_delta[index], delta_o_information_rate(single, 1)
        )
        torch.testing.assert_close(
            batch_spectral_delta[index],
            spectral_delta_o_information_rate(single, frequency, 1),
        )


def test_float32_tracks_float64_and_preserves_dtype():
    system64 = _dynamic_three(torch.float64)
    system32 = _dynamic_three(torch.float32)
    frequency64 = _grid(torch.float64, 257)
    frequency32 = frequency64.to(torch.float32)
    temporal64 = o_information_rate(system64)
    temporal32 = o_information_rate(system32)
    spectral64 = spectral_o_information_rate(system64, frequency64)
    spectral32 = spectral_o_information_rate(system32, frequency32)
    delta64 = delta_o_information_rate(system64, 0)
    delta32 = delta_o_information_rate(system32, 0)

    assert temporal32.dtype == torch.float32
    assert spectral32.dtype == torch.float32
    assert delta32.dtype == torch.float32
    torch.testing.assert_close(
        temporal32.to(torch.float64), temporal64, rtol=4e-4, atol=3e-5
    )
    torch.testing.assert_close(
        spectral32.to(torch.float64), spectral64, rtol=8e-4, atol=5e-5
    )
    torch.testing.assert_close(
        delta32.to(torch.float64), delta64, rtol=4e-4, atol=3e-5
    )


def test_log_base_scaling_matches_information_units():
    system = _dynamic_three()
    nats = o_information_rate(system, base=math.e)
    bits = o_information_rate(system, base=2.0)
    torch.testing.assert_close(bits, nats / math.log(2.0), rtol=1e-10, atol=1e-11)


def test_oir_validates_group_contracts():
    system = _dynamic_three()
    with pytest.raises(ValueError, match="at least two"):
        o_information_rate(system, groups=([0],))
    with pytest.raises(ValueError, match="pairwise disjoint"):
        o_information_rate(system, groups=([0, 1], [1, 2]))
    with pytest.raises(ValueError, match="out-of-range"):
        o_information_rate(system, groups=([0], [1], [3]))
    with pytest.raises(ValueError, match="at least three"):
        delta_o_information_rate(system, 0, groups=([0], [1]))
    with pytest.raises(ValueError, match="out of range"):
        delta_o_information_rate(system, 4)
    with pytest.raises(ValueError, match="base"):
        o_information_rate(system, base=1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_oir_cuda_matches_cpu_without_leaving_device():
    cpu = _dynamic_three(torch.float64)
    cuda = InnovationsStateSpace(
        cpu.transition.cuda(),
        cpu.observation.cuda(),
        cpu.gain.cuda(),
        cpu.innovation_covariance.cuda(),
    )
    frequency_cpu = _grid(torch.float64, 257)
    frequency_cuda = frequency_cpu.cuda()
    temporal_cuda = o_information_rate(cuda)
    spectral_cuda = spectral_o_information_rate(cuda, frequency_cuda)
    delta_cuda = delta_o_information_rate(cuda, 1)
    spectral_delta_cuda = spectral_delta_o_information_rate(
        cuda, frequency_cuda, 1
    )

    assert temporal_cuda.device.type == "cuda"
    assert spectral_cuda.device.type == "cuda"
    assert delta_cuda.device.type == "cuda"
    assert spectral_delta_cuda.device.type == "cuda"
    torch.testing.assert_close(temporal_cuda.cpu(), o_information_rate(cpu))
    torch.testing.assert_close(
        spectral_cuda.cpu(), spectral_o_information_rate(cpu, frequency_cpu)
    )
    torch.testing.assert_close(delta_cuda.cpu(), delta_o_information_rate(cpu, 1))
    torch.testing.assert_close(
        spectral_delta_cuda.cpu(),
        spectral_delta_o_information_rate(cpu, frequency_cpu, 1),
    )
