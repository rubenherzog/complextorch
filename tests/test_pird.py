import math

import pytest
import torch

import complextorch
from complextorch import (
    InnovationsStateSpace,
    build_var_system,
    gaussian_mutual_information_rate,
    partial_information_rate_decomposition,
    spectral_partial_information_rate_decomposition,
    var_to_innovations_state_space,
)
from complextorch.measures._pid_lattice import pid_lattice, pid_redundancy_from_atoms


def _iss(coefficients, covariance, *, dtype=torch.float64):
    return var_to_innovations_state_space(
        build_var_system(
            torch.as_tensor(coefficients, dtype=dtype),
            torch.as_tensor(covariance, dtype=dtype),
        )
    )


def _three_process(dtype=torch.float64):
    coefficients = torch.tensor(
        [[[0.42, 0.00, 0.00], [0.08, 0.34, 0.00], [0.30, -0.18, 0.28]]],
        dtype=dtype,
    )
    covariance = torch.tensor(
        [[1.0, 0.12, 0.04], [0.12, 0.85, -0.05], [0.04, -0.05, 0.9]],
        dtype=dtype,
    )
    return _iss(coefficients, covariance, dtype=dtype)


def _four_process(dtype=torch.float64):
    coefficients = torch.tensor(
        [[
            [0.38, 0.00, 0.00, 0.00],
            [0.04, 0.33, 0.00, 0.00],
            [0.00, -0.06, 0.29, 0.00],
            [0.26, 0.17, -0.14, 0.25],
        ]],
        dtype=dtype,
    )
    covariance = torch.tensor(
        [
            [1.0, 0.10, 0.02, 0.04],
            [0.10, 0.9, -0.04, 0.02],
            [0.02, -0.04, 0.8, -0.03],
            [0.04, 0.02, -0.03, 0.85],
        ],
        dtype=dtype,
    )
    return _iss(coefficients, covariance, dtype=dtype)


def _grid(dtype=torch.float64, n=1025, sampling_frequency=1.0):
    return torch.linspace(0.0, sampling_frequency / 2.0, n, dtype=dtype)


def _subset_position(result, subset):
    return result.source_subsets.index(frozenset(subset))


def _antichain_position(result, antichain):
    target = frozenset(frozenset(subset) for subset in antichain)
    return next(
        index
        for index, node in enumerate(result.antichains)
        if frozenset(node) == target
    )


def _direct_subset_rates(system, sources, target):
    subsets = tuple(
        frozenset(index for index in range(len(sources)) if mask & (1 << index))
        for mask in range(1, 1 << len(sources))
    )
    rates = []
    for subset in subsets:
        indices = tuple(
            observation
            for source_index in sorted(subset)
            for observation in sources[source_index]
        )
        rates.append(gaussian_mutual_information_rate(system, indices, target))
    return subsets, torch.stack(rates, dim=-1)


def test_public_pird_api_is_exported_and_documented():
    names = (
        "PIRDResult",
        "SpectralPIRDResult",
        "partial_information_rate_decomposition",
        "spectral_partial_information_rate_decomposition",
    )
    for name in names:
        assert name in complextorch.__all__
        assert getattr(complextorch, name).__doc__


def test_two_source_pird_matches_explicit_minimum_information_formulas():
    result = spectral_partial_information_rate_decomposition(
        _three_process(), ([0], [1]), [2], _grid(n=513)
    )
    i0 = result.subset_mir[..., _subset_position(result, {0}), :]
    i1 = result.subset_mir[..., _subset_position(result, {1}), :]
    i01 = result.subset_mir[..., _subset_position(result, {0, 1}), :]
    redundancy = torch.minimum(i0, i1)
    unique0 = i0 - redundancy
    unique1 = i1 - redundancy
    synergy = i01 - redundancy - unique0 - unique1

    torch.testing.assert_close(result.redundant, redundancy, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(result.unique[..., 0, :], unique0, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(result.unique[..., 1, :], unique1, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(result.synergistic, synergy, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(result.delta, redundancy - synergy, rtol=0.0, atol=1e-12)

    expected_atoms = {
        ((0,), (1,)): redundancy,
        ((0,),): unique0,
        ((1,),): unique1,
        ((0, 1),): synergy,
    }
    for antichain, expected in expected_atoms.items():
        position = _antichain_position(result, antichain)
        torch.testing.assert_close(result.atoms[..., position, :], expected, rtol=0.0, atol=1e-12)


def _manual_three_source_atoms(result):
    """Independent 18-node oracle following the handwritten PIRD recursion."""
    rate = {
        subset: result.subset_mir[..., index, :]
        for index, subset in enumerate(result.source_subsets)
    }
    rows = (
        ((0,), (1,), (2,)),
        ((0,), (1,)),
        ((0,), (2,)),
        ((1,), (2,)),
        ((0,), (1, 2)),
        ((1,), (0, 2)),
        ((0, 1), (2,)),
        ((0,),),
        ((1,),),
        ((2,),),
        ((0, 1), (0, 2), (1, 2)),
        ((0, 1), (0, 2)),
        ((0, 1), (1, 2)),
        ((0, 2), (1, 2)),
        ((0, 1),),
        ((0, 2),),
        ((1, 2),),
        ((0, 1, 2),),
    )
    r = []
    for antichain in rows:
        terms = torch.stack([rate[frozenset(subset)] for subset in antichain], dim=-1)
        r.append(terms.amin(dim=-1))
    atoms = [None] * 18
    atoms[0] = r[0]
    atoms[1] = r[1] - atoms[0]
    atoms[2] = r[2] - atoms[0]
    atoms[3] = r[3] - atoms[0]
    atoms[4] = r[4] - sum(atoms[index] for index in (0, 1, 2))
    atoms[5] = r[5] - sum(atoms[index] for index in (0, 1, 3))
    atoms[6] = r[6] - sum(atoms[index] for index in (0, 2, 3))
    atoms[7] = r[7] - sum(atoms[index] for index in (0, 1, 2, 4))
    atoms[8] = r[8] - sum(atoms[index] for index in (0, 1, 3, 5))
    atoms[9] = r[9] - sum(atoms[index] for index in (0, 2, 3, 6))
    atoms[10] = r[10] - sum(atoms[index] for index in (0, 1, 2, 3, 4, 5, 6))
    atoms[11] = r[11] - sum(atoms[index] for index in (0, 1, 2, 3, 4, 5, 6, 7, 10))
    atoms[12] = r[12] - sum(atoms[index] for index in (0, 1, 2, 3, 4, 5, 6, 8, 10))
    atoms[13] = r[13] - sum(atoms[index] for index in (0, 1, 2, 3, 4, 5, 6, 9, 10))
    atoms[14] = r[14] - sum(atoms[index] for index in (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12))
    atoms[15] = r[15] - sum(atoms[index] for index in (0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13))
    atoms[16] = r[16] - sum(atoms[index] for index in (0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13))
    atoms[17] = r[17] - sum(atoms[:17])
    return rows, atoms


def test_three_source_generic_mobius_matches_handwritten_pird_oracle():
    result = spectral_partial_information_rate_decomposition(
        _four_process(), ([0], [1], [2]), [3], _grid(n=257)
    )
    rows, manual_atoms = _manual_three_source_atoms(result)
    assert len(result.antichains) == 18
    for antichain, expected in zip(rows, manual_atoms, strict=True):
        position = _antichain_position(result, antichain)
        torch.testing.assert_close(result.atoms[..., position, :], expected, rtol=0.0, atol=2e-12)


def test_every_temporal_subset_mir_is_reconstructed_from_integrated_atoms():
    system = _four_process()
    sources = ((0,), (1,), (2,))
    result = partial_information_rate_decomposition(
        system, sources, [3], _grid(n=2049)
    )
    subsets, direct = _direct_subset_rates(system, sources, (3,))
    lattice = pid_lattice(3, device=result.atoms.device)
    reconstructed = pid_redundancy_from_atoms(result.atoms, lattice)
    for subset_index, subset in enumerate(subsets):
        position = lattice.index((tuple(sorted(subset)),))
        torch.testing.assert_close(result.subset_mir[..., subset_index], direct[..., subset_index], rtol=5e-7, atol=6e-9)
        torch.testing.assert_close(reconstructed[..., position], direct[..., subset_index], rtol=5e-7, atol=6e-9)
    joint_position = subsets.index(frozenset({0, 1, 2}))
    torch.testing.assert_close(result.atoms.sum(dim=-1), direct[..., joint_position], rtol=5e-7, atol=6e-9)


def test_spectral_atoms_reconstruct_every_redundancy_function_pointwise():
    result = spectral_partial_information_rate_decomposition(
        _four_process(), ([0], [1], [2]), [3], _grid(n=129)
    )
    lattice = pid_lattice(3, device=result.atoms.device)
    reconstructed = pid_redundancy_from_atoms(result.atoms.movedim(-2, -1), lattice)
    torch.testing.assert_close(reconstructed.movedim(-1, -2), result.redundancy, rtol=0.0, atol=2e-12)


def test_coarse_graining_conserves_joint_mir_for_two_and_three_sources():
    frequency = _grid(n=1025)
    cases = (
        (_three_process(), ((0,), (1,)), (2,)),
        (_four_process(), ((0,), (1,), (2,)), (3,)),
    )
    for system, sources, target in cases:
        result = partial_information_rate_decomposition(system, sources, target, frequency)
        joint_subset = frozenset(range(len(sources)))
        joint = result.subset_mir[..., result.source_subsets.index(joint_subset)]
        coarse = result.unique.sum(dim=-1) + result.redundant + result.synergistic
        torch.testing.assert_close(coarse, joint, rtol=5e-7, atol=6e-9)
        torch.testing.assert_close(result.atoms.sum(dim=-1), joint, rtol=5e-7, atol=6e-9)


def test_iid_examples_have_expected_redundancy_vs_synergy_balance():
    zero = torch.zeros((1, 3, 3), dtype=torch.float64)
    redundant_covariance = torch.tensor(
        [[1.2, 0.8, 0.8], [0.8, 1.2, 0.8], [0.8, 0.8, 1.2]], dtype=torch.float64
    )
    synergistic_covariance = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 2.5]], dtype=torch.float64
    )
    frequency = _grid(n=65)
    redundant = partial_information_rate_decomposition(
        _iss(zero, redundant_covariance), ([0], [1]), [2], frequency
    )
    synergistic = partial_information_rate_decomposition(
        _iss(zero, synergistic_covariance), ([0], [1]), [2], frequency
    )
    assert bool((redundant.delta > 0).all().item())
    assert bool((synergistic.delta < 0).all().item())


def test_source_permutation_preserves_invariant_terms_and_permutes_uniques():
    system = _four_process()
    frequency = _grid(n=513)
    original = partial_information_rate_decomposition(
        system, ([0], [1], [2]), [3], frequency
    )
    permuted = partial_information_rate_decomposition(
        system, ([2], [0], [1]), [3], frequency
    )
    torch.testing.assert_close(permuted.redundant, original.redundant, rtol=3e-9, atol=3e-10)
    torch.testing.assert_close(permuted.synergistic, original.synergistic, rtol=3e-9, atol=3e-10)
    torch.testing.assert_close(permuted.delta, original.delta, rtol=3e-9, atol=3e-10)
    torch.testing.assert_close(permuted.unique[..., 0], original.unique[..., 2], rtol=3e-9, atol=3e-10)
    torch.testing.assert_close(permuted.unique[..., 1], original.unique[..., 0], rtol=3e-9, atol=3e-10)
    torch.testing.assert_close(permuted.unique[..., 2], original.unique[..., 1], rtol=3e-9, atol=3e-10)


def test_faes_half_open_integration_converges_to_temporal_subset_mirs():
    system = _three_process()
    _, direct = _direct_subset_rates(system, ((0,), (1,)), (2,))
    errors = []
    fine_result = None
    for n in (512, 4096):
        frequency = torch.arange(n, dtype=torch.float64) / (2.0 * n)
        result = partial_information_rate_decomposition(
            system, ([0], [1]), [2], frequency, half_open=True
        )
        errors.append(torch.abs(result.subset_mir - direct))
        fine_result = result
    assert bool(torch.all(errors[1] < errors[0]).item())
    assert float(errors[1].max().item()) < 2.0e-5
    torch.testing.assert_close(fine_result.subset_mir, direct, rtol=4e-4, atol=2e-6)


def test_batched_pird_matches_explicit_loop_for_all_outputs():
    coefficient = torch.tensor(
        [
            [[0.40, 0.00, 0.00], [0.05, 0.32, 0.00], [0.22, 0.11, 0.27]],
            [[0.34, -0.04, 0.00], [0.08, 0.36, 0.00], [0.15, -0.12, 0.30]],
            [[0.30, 0.03, 0.00], [-0.02, 0.28, 0.00], [0.18, 0.14, 0.33]],
        ], dtype=torch.float64,
    ).unsqueeze(1)
    covariance = torch.tensor(
        [
            [[1.0, 0.08, 0.03], [0.08, 0.9, -0.02], [0.03, -0.02, 0.8]],
            [[0.9, -0.05, 0.02], [-0.05, 1.1, 0.04], [0.02, 0.04, 0.85]],
            [[1.2, 0.04, -0.03], [0.04, 0.8, 0.05], [-0.03, 0.05, 0.95]],
        ], dtype=torch.float64,
    )
    system = _iss(coefficient, covariance)
    frequency = _grid(n=129)
    spectral = spectral_partial_information_rate_decomposition(
        system, ([0], [1]), [2], frequency
    )
    temporal = partial_information_rate_decomposition(
        system, ([0], [1]), [2], frequency
    )
    assert spectral.atoms.shape[:2] == (3, 4)
    assert temporal.atoms.shape == (3, 4)
    assert temporal.unique.shape == (3, 2)

    fields = ("subset_mir", "redundancy", "atoms", "unique", "redundant", "synergistic", "delta")
    for batch in range(3):
        single = InnovationsStateSpace(
            system.transition[batch], system.observation[batch], system.gain[batch], system.innovation_covariance[batch]
        )
        single_spectral = spectral_partial_information_rate_decomposition(
            single, ([0], [1]), [2], frequency
        )
        single_temporal = partial_information_rate_decomposition(
            single, ([0], [1]), [2], frequency
        )
        for field in fields:
            torch.testing.assert_close(getattr(spectral, field)[batch], getattr(single_spectral, field))
            torch.testing.assert_close(getattr(temporal, field)[batch], getattr(single_temporal, field))


def test_grouped_sources_are_supported_without_reshaping_channels():
    dtype = torch.float64
    coefficients = torch.tensor(
        [[
            [0.25, 0.00, 0.00, 0.00, 0.00],
            [0.05, 0.28, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.30, 0.00, 0.00],
            [0.00, 0.00, 0.04, 0.27, 0.00],
            [0.16, 0.10, 0.13, -0.08, 0.22],
        ]], dtype=dtype,
    )
    result = partial_information_rate_decomposition(
        _iss(coefficients, torch.eye(5, dtype=dtype)),
        ([0, 1], [2, 3]),
        [4],
        _grid(n=513),
    )
    assert result.sources == ((0, 1), (2, 3))
    assert result.target == (4,)
    assert result.unique.shape[-1] == 2


def test_float32_tracks_float64_and_preserves_dtype():
    result64 = partial_information_rate_decomposition(
        _three_process(torch.float64), ([0], [1]), [2], _grid(torch.float64, 257)
    )
    result32 = partial_information_rate_decomposition(
        _three_process(torch.float32), ([0], [1]), [2], _grid(torch.float32, 257)
    )
    fields = ("subset_mir", "redundancy", "atoms", "unique", "redundant", "synergistic", "delta")
    for field in fields:
        value32 = getattr(result32, field)
        value64 = getattr(result64, field)
        assert value32.dtype == torch.float32
        torch.testing.assert_close(value32.to(torch.float64), value64, rtol=8e-4, atol=6e-5)


def test_log_base_scales_all_information_atoms():
    system = _three_process()
    frequency = _grid(n=257)
    nats = partial_information_rate_decomposition(system, ([0], [1]), [2], frequency, base=math.e)
    bits = partial_information_rate_decomposition(system, ([0], [1]), [2], frequency, base=2.0)
    torch.testing.assert_close(bits.atoms, nats.atoms / math.log(2.0), rtol=2e-9, atol=2e-10)


def test_pird_validates_source_target_contracts():
    system = _four_process()
    frequency = _grid(n=65)
    with pytest.raises(ValueError, match="two or three"):
        spectral_partial_information_rate_decomposition(system, ([0],), [3], frequency)
    with pytest.raises(ValueError, match="two or three"):
        spectral_partial_information_rate_decomposition(system, ([0], [1], [2], [3]), [3], frequency)
    with pytest.raises(ValueError, match="disjoint"):
        spectral_partial_information_rate_decomposition(system, ([0], [1]), [1], frequency)
    with pytest.raises(ValueError, match="pairwise disjoint"):
        spectral_partial_information_rate_decomposition(system, ([0, 1], [1, 2]), [3], frequency)
    with pytest.raises(ValueError, match="base"):
        partial_information_rate_decomposition(system, ([0], [1]), [3], frequency, base=1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_pird_cuda_matches_cpu_and_stays_on_device():
    cpu = _three_process(torch.float64)
    cuda = InnovationsStateSpace(
        cpu.transition.cuda(), cpu.observation.cuda(), cpu.gain.cuda(), cpu.innovation_covariance.cuda()
    )
    result_cpu = partial_information_rate_decomposition(
        cpu, ([0], [1]), [2], _grid(torch.float64, 257)
    )
    result_cuda = partial_information_rate_decomposition(
        cuda, ([0], [1]), [2], _grid(torch.float64, 257).cuda()
    )
    fields = ("subset_mir", "redundancy", "atoms", "unique", "redundant", "synergistic", "delta")
    for field in fields:
        value_cuda = getattr(result_cuda, field)
        assert value_cuda.device.type == "cuda"
        torch.testing.assert_close(value_cuda.cpu(), getattr(result_cpu, field), rtol=2e-9, atol=2e-10)
