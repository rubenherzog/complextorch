import pytest
import torch

import complextorch
from complextorch import (
    build_var_system,
    partial_granger_causality_decomposition,
    spectral_partial_granger_causality_decomposition,
    var_to_innovations_state_space,
)
from complextorch.measures._pid_lattice import pid_lattice, pid_redundancy_from_atoms
from complextorch.measures.mvgc import state_space_temporal_mvgc


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


def test_public_pdgc_api_is_exported_and_documented():
    names = (
        "PDGCResult",
        "SpectralPDGCResult",
        "partial_granger_causality_decomposition",
        "spectral_partial_granger_causality_decomposition",
    )
    for name in names:
        assert name in complextorch.__all__
        assert getattr(complextorch, name).__doc__


def test_two_source_pdgc_matches_explicit_minimum_gc_formulas():
    result = spectral_partial_granger_causality_decomposition(
        _three_process(), ([0], [1]), [2], _grid(n=513)
    )
    g0 = result.subset_gc[..., _subset_position(result, {0}), :]
    g1 = result.subset_gc[..., _subset_position(result, {1}), :]
    g01 = result.subset_gc[..., _subset_position(result, {0, 1}), :]
    redundancy = torch.minimum(g0, g1)
    unique0 = g0 - redundancy
    unique1 = g1 - redundancy
    synergy = g01 - redundancy - unique0 - unique1

    torch.testing.assert_close(result.redundant, redundancy, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(result.unique[..., 0, :], unique0, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(result.unique[..., 1, :], unique1, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(result.synergistic, synergy, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(result.delta, redundancy - synergy, rtol=0.0, atol=1e-12)


def test_three_source_atoms_reconstruct_all_redundancy_functions_pointwise():
    result = spectral_partial_granger_causality_decomposition(
        _four_process(), ([0], [1], [2]), [3], _grid(n=257)
    )
    lattice = pid_lattice(3, device=result.atoms.device)
    reconstructed = pid_redundancy_from_atoms(result.atoms.movedim(-2, -1), lattice)
    torch.testing.assert_close(
        reconstructed.movedim(-1, -2), result.redundancy, rtol=0.0, atol=3e-12
    )
    assert len(result.antichains) == 18


def test_integrated_subset_gc_matches_exact_temporal_state_space_gc():
    system = _four_process()
    sources = ((0,), (1,), (2,))
    result = partial_granger_causality_decomposition(
        system, sources, (3,), _grid(n=2049)
    )
    lattice = pid_lattice(3, device=result.atoms.device)
    reconstructed = pid_redundancy_from_atoms(result.atoms, lattice)
    for subset_index, subset in enumerate(result.source_subsets):
        source_indices = tuple(
            observation
            for source_position in sorted(subset)
            for observation in sources[source_position]
        )
        direct = state_space_temporal_mvgc(
            system, source=source_indices, target=(3,), conditional=()
        )
        torch.testing.assert_close(
            result.subset_gc[..., subset_index], direct, rtol=8e-7, atol=8e-9
        )
        node = lattice.index((tuple(sorted(subset)),))
        torch.testing.assert_close(
            reconstructed[..., node], direct, rtol=8e-7, atol=8e-9
        )


def test_total_atoms_and_coarse_graining_conserve_joint_gc():
    for system, sources, target in (
        (_three_process(), ((0,), (1,)), (2,)),
        (_four_process(), ((0,), (1,), (2,)), (3,)),
    ):
        result = partial_granger_causality_decomposition(
            system, sources, target, _grid(n=1025)
        )
        joint_subset = frozenset(range(len(sources)))
        joint = result.subset_gc[..., result.source_subsets.index(joint_subset)]
        coarse = result.unique.sum(dim=-1) + result.redundant + result.synergistic
        torch.testing.assert_close(result.atoms.sum(dim=-1), joint, rtol=8e-7, atol=8e-9)
        torch.testing.assert_close(coarse, joint, rtol=8e-7, atol=8e-9)


def test_uncoupled_sources_have_zero_pdgc():
    coefficients = torch.tensor(
        [[[0.35, 0.0, 0.0], [0.0, 0.30, 0.0], [0.0, 0.0, 0.25]]],
        dtype=torch.float64,
    )
    result = partial_granger_causality_decomposition(
        _iss(coefficients, torch.eye(3)), ((0,), (1,)), (2,), _grid(n=257)
    )
    torch.testing.assert_close(
        result.subset_gc, torch.zeros_like(result.subset_gc), atol=2e-12, rtol=0.0
    )
    torch.testing.assert_close(
        result.atoms, torch.zeros_like(result.atoms), atol=2e-12, rtol=0.0
    )


def test_source_permutation_preserves_invariant_terms_and_permutes_uniques():
    system = _four_process()
    frequency = _grid(n=513)
    original = partial_granger_causality_decomposition(
        system, ((0,), (1,), (2,)), (3,), frequency
    )
    permuted = partial_granger_causality_decomposition(
        system, ((2,), (0,), (1,)), (3,), frequency
    )
    torch.testing.assert_close(permuted.redundant, original.redundant, rtol=3e-9, atol=3e-10)
    torch.testing.assert_close(permuted.synergistic, original.synergistic, rtol=3e-9, atol=3e-10)
    torch.testing.assert_close(permuted.delta, original.delta, rtol=3e-9, atol=3e-10)
    torch.testing.assert_close(permuted.unique[..., 0], original.unique[..., 2], rtol=3e-9, atol=3e-10)
    torch.testing.assert_close(permuted.unique[..., 1], original.unique[..., 0], rtol=3e-9, atol=3e-10)
    torch.testing.assert_close(permuted.unique[..., 2], original.unique[..., 1], rtol=3e-9, atol=3e-10)


def test_grouped_multichannel_sources_are_supported():
    coefficients = torch.zeros((1, 5, 5), dtype=torch.float64)
    coefficients[0].diagonal().copy_(torch.tensor([0.3, 0.25, 0.2, 0.15, 0.22]))
    coefficients[0, 4, 0] = 0.20
    coefficients[0, 4, 1] = -0.12
    coefficients[0, 4, 2] = 0.16
    result = spectral_partial_granger_causality_decomposition(
        _iss(coefficients, torch.eye(5)), ((0, 1), (2, 3)), (4,), _grid(n=65)
    )
    assert result.subset_gc.shape == (3, 65)
    assert result.unique.shape == (2, 65)


def test_faes_half_open_subset_gc_converges_to_temporal_gc():
    system = _three_process()
    direct = torch.stack(
        [
            state_space_temporal_mvgc(system, source=(0,), target=(2,), conditional=()),
            state_space_temporal_mvgc(system, source=(1,), target=(2,), conditional=()),
            state_space_temporal_mvgc(system, source=(0, 1), target=(2,), conditional=()),
        ],
        dim=-1,
    )
    errors = []
    for n in (256, 2048):
        frequency = torch.arange(n, dtype=torch.float64) / (2.0 * n)
        result = partial_granger_causality_decomposition(
            system, ((0,), (1,)), (2,), frequency, half_open=True
        )
        errors.append((result.subset_gc - direct).abs().max())
    assert float(errors[1]) < float(errors[0])
    assert float(errors[1]) < 2e-4


def test_batched_pdgc_matches_explicit_loop_for_all_outputs():
    coefficient = torch.tensor(
        [
            [[0.40, 0.00, 0.00], [0.05, 0.32, 0.00], [0.22, 0.11, 0.27]],
            [[0.34, -0.04, 0.00], [0.08, 0.36, 0.00], [0.15, -0.12, 0.30]],
            [[0.30, 0.03, 0.00], [-0.02, 0.31, 0.00], [0.19, 0.08, 0.24]],
        ],
        dtype=torch.float64,
    ).unsqueeze(1)
    covariance = torch.stack(
        [
            torch.tensor([[1.0, 0.05, 0.02], [0.05, 0.9, 0.01], [0.02, 0.01, 0.8]]),
            torch.tensor([[0.9, -0.04, 0.01], [-0.04, 1.1, 0.03], [0.01, 0.03, 0.85]]),
            torch.tensor([[1.2, 0.02, -0.03], [0.02, 0.95, 0.04], [-0.03, 0.04, 0.9]]),
        ]
    ).to(torch.float64)
    system = _iss(coefficient, covariance)
    frequency = _grid(n=129)
    batched_spectral = spectral_partial_granger_causality_decomposition(
        system, ((0,), (1,)), (2,), frequency
    )
    batched_temporal = partial_granger_causality_decomposition(
        system, ((0,), (1,)), (2,), frequency
    )
    for batch in range(3):
        single_system = _iss(coefficient[batch], covariance[batch])
        single_spectral = spectral_partial_granger_causality_decomposition(
            single_system, ((0,), (1,)), (2,), frequency
        )
        single_temporal = partial_granger_causality_decomposition(
            single_system, ((0,), (1,)), (2,), frequency
        )
        for name in (
            "subset_gc", "redundancy", "atoms", "unique",
            "redundant", "synergistic", "delta",
        ):
            torch.testing.assert_close(
                getattr(batched_spectral, name)[batch], getattr(single_spectral, name)
            )
            torch.testing.assert_close(
                getattr(batched_temporal, name)[batch], getattr(single_temporal, name)
            )


def test_pdgc_float32_agrees_with_float64():
    result64 = partial_granger_causality_decomposition(
        _three_process(torch.float64), ((0,), (1,)), (2,), _grid(n=129)
    )
    result32 = partial_granger_causality_decomposition(
        _three_process(torch.float32),
        ((0,), (1,)),
        (2,),
        _grid(torch.float32, n=129),
    )
    assert result32.atoms.dtype == torch.float32
    for name in ("subset_gc", "atoms", "unique", "redundant", "synergistic", "delta"):
        torch.testing.assert_close(
            getattr(result32, name).to(torch.float64),
            getattr(result64, name),
            rtol=3e-4,
            atol=3e-5,
        )


def test_invalid_pdgc_groups_and_frequency_are_rejected():
    system = _three_process()
    with pytest.raises(ValueError):
        spectral_partial_granger_causality_decomposition(system, ((0,),), (2,), _grid(n=9))
    with pytest.raises(ValueError):
        spectral_partial_granger_causality_decomposition(system, ((0,), (1,)), (1,), _grid(n=9))
    with pytest.raises(ValueError):
        spectral_partial_granger_causality_decomposition(
            system, ((0,), (1,)), (2,), torch.tensor([0.0, 0.6])
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_pdgc_cuda_matches_cpu():
    cpu_system = _three_process()
    gpu_system = complextorch.InnovationsStateSpace(
        transition=cpu_system.transition.cuda(),
        observation=cpu_system.observation.cuda(),
        gain=cpu_system.gain.cuda(),
        innovation_covariance=cpu_system.innovation_covariance.cuda(),
    )
    frequency = _grid(n=129)
    cpu = partial_granger_causality_decomposition(
        cpu_system, ((0,), (1,)), (2,), frequency
    )
    gpu = partial_granger_causality_decomposition(
        gpu_system, ((0,), (1,)), (2,), frequency.cuda()
    )
    for name in ("subset_gc", "redundancy", "atoms", "unique", "redundant", "synergistic", "delta"):
        torch.testing.assert_close(
            getattr(gpu, name).cpu(), getattr(cpu, name), rtol=2e-9, atol=2e-10
        )
