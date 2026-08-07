import pytest
import torch

import complextorch
from complextorch import (
    build_var_system,
    hop_analysis,
    partial_granger_causality_decomposition,
    partial_information_rate_decomposition,
    spectral_hop_analysis,
    spectral_partial_granger_causality_decomposition,
    spectral_partial_information_rate_decomposition,
    var_to_innovations_state_space,
)


def _var(dtype=torch.float64, *, batch=False):
    coefficient = torch.tensor(
        [[[0.36, 0.00, 0.00], [0.05, 0.31, 0.00], [0.22, -0.12, 0.27]]],
        dtype=dtype,
    )
    covariance = torch.tensor(
        [[1.0, 0.06, 0.02], [0.06, 0.9, -0.01], [0.02, -0.01, 0.8]],
        dtype=dtype,
    )
    if not batch:
        return build_var_system(coefficient, covariance)
    coefficient = torch.stack(
        [
            coefficient,
            torch.tensor(
                [[[0.31, 0.02, 0.00], [-0.04, 0.34, 0.00], [0.17, 0.09, 0.29]]],
                dtype=dtype,
            ),
        ]
    )
    covariance = torch.stack(
        [
            covariance,
            torch.tensor(
                [[0.9, -0.03, 0.01], [-0.03, 1.1, 0.04], [0.01, 0.04, 0.85]],
                dtype=dtype,
            ),
        ]
    )
    return build_var_system(coefficient, covariance)


def _grid(dtype=torch.float64, n=129):
    return torch.linspace(0.0, 0.5, n, dtype=dtype)


def _assert_component_equal(left, right, names):
    for name in names:
        torch.testing.assert_close(getattr(left, name), getattr(right, name))


def test_hop_public_api_is_exported():
    assert complextorch.HOPResult is not None
    assert complextorch.SpectralHOPResult is not None
    assert complextorch.hop_analysis is hop_analysis
    assert complextorch.spectral_hop_analysis is spectral_hop_analysis


def test_spectral_hop_is_exact_composition_of_pird_and_pdgc():
    system = var_to_innovations_state_space(_var())
    frequency = _grid()
    hop = spectral_hop_analysis(system, ((0,), (1,)), (2,), frequency)
    pird = spectral_partial_information_rate_decomposition(
        system, ((0,), (1,)), (2,), frequency
    )
    pdgc = spectral_partial_granger_causality_decomposition(
        system, ((0,), (1,)), (2,), frequency
    )
    assert hop.sources == pird.sources == pdgc.sources
    assert hop.target == pird.target == pdgc.target
    assert pird.source_subsets == pdgc.source_subsets
    assert pird.antichains == pdgc.antichains
    torch.testing.assert_close(hop.frequencies, frequency)
    _assert_component_equal(
        hop.pird,
        pird,
        ("subset_mir", "redundancy", "atoms", "unique", "redundant", "synergistic", "delta"),
    )
    _assert_component_equal(
        hop.pdgc,
        pdgc,
        ("subset_gc", "redundancy", "atoms", "unique", "redundant", "synergistic", "delta"),
    )


def test_integrated_hop_is_exact_composition_of_pird_and_pdgc():
    system = var_to_innovations_state_space(_var())
    frequency = _grid()
    hop = hop_analysis(system, ((0,), (1,)), (2,), frequency)
    pird = partial_information_rate_decomposition(system, ((0,), (1,)), (2,), frequency)
    pdgc = partial_granger_causality_decomposition(system, ((0,), (1,)), (2,), frequency)
    _assert_component_equal(
        hop.pird,
        pird,
        ("subset_mir", "redundancy", "atoms", "unique", "redundant", "synergistic", "delta"),
    )
    _assert_component_equal(
        hop.pdgc,
        pdgc,
        ("subset_gc", "redundancy", "atoms", "unique", "redundant", "synergistic", "delta"),
    )


def test_hop_accepts_var_and_matches_exact_innovations_conversion():
    system = _var()
    innovations = var_to_innovations_state_space(system)
    frequency = _grid()
    from_var = hop_analysis(system, ((0,), (1,)), (2,), frequency)
    from_iss = hop_analysis(innovations, ((0,), (1,)), (2,), frequency)
    torch.testing.assert_close(from_var.pird.atoms, from_iss.pird.atoms)
    torch.testing.assert_close(from_var.pdgc.atoms, from_iss.pdgc.atoms)


def test_hop_half_open_is_forwarded_identically_to_both_components():
    system = var_to_innovations_state_space(_var())
    n = 512
    frequency = torch.arange(n, dtype=torch.float64) / (2.0 * n)
    hop = hop_analysis(system, ((0,), (1,)), (2,), frequency, half_open=True)
    pird = partial_information_rate_decomposition(
        system, ((0,), (1,)), (2,), frequency, half_open=True
    )
    pdgc = partial_granger_causality_decomposition(
        system, ((0,), (1,)), (2,), frequency, half_open=True
    )
    torch.testing.assert_close(hop.pird.atoms, pird.atoms)
    torch.testing.assert_close(hop.pdgc.atoms, pdgc.atoms)


def test_three_source_hop_uses_same_eighteen_node_lattice():
    coefficient = torch.zeros((1, 4, 4), dtype=torch.float64)
    coefficient[0].diagonal().copy_(torch.tensor([0.32, 0.28, 0.24, 0.30]))
    coefficient[0, 3, :3] = torch.tensor([0.18, -0.13, 0.11])
    system = var_to_innovations_state_space(build_var_system(coefficient, torch.eye(4)))
    result = spectral_hop_analysis(system, ((0,), (1,), (2,)), (3,), _grid(n=65))
    assert len(result.pird.antichains) == 18
    assert result.pird.antichains == result.pdgc.antichains
    assert result.pird.unique.shape == (1, 3, 65)
    assert result.pdgc.unique.shape == (1, 3, 65)


def test_grouped_multichannel_hop_preserves_group_semantics():
    coefficient = torch.zeros((1, 5, 5), dtype=torch.float64)
    coefficient[0].diagonal().copy_(torch.tensor([0.30, 0.25, 0.20, 0.15, 0.22]))
    coefficient[0, 4, 0] = 0.20
    coefficient[0, 4, 1] = -0.12
    coefficient[0, 4, 2] = 0.16
    system = var_to_innovations_state_space(build_var_system(coefficient, torch.eye(5)))
    result = spectral_hop_analysis(system, ((0, 1), (2, 3)), (4,), _grid(n=65))
    assert result.sources == ((0, 1), (2, 3))
    assert result.target == (4,)
    assert result.pird.subset_mir.shape == (1, 3, 65)
    assert result.pdgc.subset_gc.shape == (1, 3, 65)


def test_batched_hop_matches_explicit_system_loop():
    system = _var(batch=True)
    frequency = _grid(n=65)
    batched = hop_analysis(system, ((0,), (1,)), (2,), frequency)
    for index in range(2):
        single = build_var_system(
            system.coefficients[index], system.innovation_covariance[index]
        )
        loop = hop_analysis(single, ((0,), (1,)), (2,), frequency)
        for component, names in (
            ("pird", ("subset_mir", "atoms", "unique", "redundant", "synergistic", "delta")),
            ("pdgc", ("subset_gc", "atoms", "unique", "redundant", "synergistic", "delta")),
        ):
            for name in names:
                torch.testing.assert_close(
                    getattr(getattr(batched, component), name)[index],
                    getattr(getattr(loop, component), name)[0],
                )


def test_hop_preserves_dtype_and_float32_agrees_with_float64():
    result64 = hop_analysis(_var(torch.float64), ((0,), (1,)), (2,), _grid(torch.float64))
    result32 = hop_analysis(_var(torch.float32), ((0,), (1,)), (2,), _grid(torch.float32))
    assert result32.pird.atoms.dtype == torch.float32
    assert result32.pdgc.atoms.dtype == torch.float32
    torch.testing.assert_close(
        result32.pird.delta.to(torch.float64), result64.pird.delta, rtol=2e-4, atol=2e-5
    )
    torch.testing.assert_close(
        result32.pdgc.delta.to(torch.float64), result64.pdgc.delta, rtol=3e-4, atol=3e-5
    )


def test_hop_rejects_invalid_system_type():
    with pytest.raises(TypeError, match="VARSystem or InnovationsStateSpace"):
        hop_analysis(torch.eye(3), ((0,), (1,)), (2,), _grid())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_hop_cuda_matches_cpu():
    cpu = hop_analysis(_var(), ((0,), (1,)), (2,), _grid(n=65))
    system = _var()
    gpu = build_var_system(
        system.coefficients.cuda(), system.innovation_covariance.cuda()
    )
    result = hop_analysis(gpu, ((0,), (1,)), (2,), _grid(n=65).cuda())
    torch.testing.assert_close(result.pird.atoms.cpu(), cpu.pird.atoms, rtol=1e-8, atol=1e-9)
    torch.testing.assert_close(result.pdgc.atoms.cpu(), cpu.pdgc.atoms, rtol=1e-8, atol=1e-9)
