import math

import torch

from complextorch import (
    build_var_system,
    gaussian_mutual_information_rate,
    innovations_spectral_density,
    integrate_spectral_rate,
    reduce_innovations_state_space,
    solve_generalized_dare,
    spectral_gaussian_mutual_information_rate,
    var_to_innovations_state_space,
    varma_to_innovations_state_space,
)
from complextorch.linalg import spd_solve, symmetrise
from complextorch.measures._pid_lattice import (
    pid_lattice,
    pid_mobius_inversion,
    pid_redundancy_from_atoms,
)
from complextorch.measures.dynamics import cross_spectral_density
from complextorch.measures.gaussian import o_information
from complextorch.measures.mvgc import (
    state_space_spectral_mvgc,
    state_space_temporal_mvgc,
)


def _three_variable_var():
    dtype = torch.float64
    coefficients = torch.tensor(
        [
            [
                [0.45, 0.12, 0.00],
                [0.05, 0.36, 0.08],
                [0.18, -0.10, 0.30],
            ],
            [
                [0.08, 0.00, 0.00],
                [0.00, -0.06, 0.00],
                [0.04, 0.03, 0.05],
            ],
        ],
        dtype=dtype,
    )
    covariance = torch.tensor(
        [
            [1.00, 0.22, 0.08],
            [0.22, 0.85, -0.05],
            [0.08, -0.05, 0.70],
        ],
        dtype=dtype,
    )
    return build_var_system(coefficients, covariance)


def _four_variable_iss():
    dtype = torch.float64
    coefficients = torch.tensor(
        [
            [
                [0.36, 0.08, 0.00, 0.03],
                [0.04, 0.31, 0.06, 0.00],
                [0.02, -0.05, 0.29, 0.07],
                [0.12, 0.09, -0.08, 0.34],
            ]
        ],
        dtype=dtype,
    )
    covariance = torch.tensor(
        [
            [1.00, 0.18, 0.05, 0.03],
            [0.18, 0.90, -0.04, 0.06],
            [0.05, -0.04, 0.80, 0.10],
            [0.03, 0.06, 0.10, 0.75],
        ],
        dtype=dtype,
    )
    return var_to_innovations_state_space(build_var_system(coefficients, covariance))


def _grid(n: int = 1025):
    return torch.linspace(0.0, 0.5, n, dtype=torch.float64)


def _spectral_redundancy(lattice, subset_rates):
    values = []
    for antichain in lattice.antichains:
        components = torch.stack(
            [subset_rates[frozenset(subset)] for subset in antichain], dim=-1
        )
        values.append(components.min(dim=-1).values)
    return torch.stack(values, dim=-1)


def test_f01_var_and_innovations_state_space_spectra_are_identical():
    """F01: VAR and exact companion innovations forms have the same spectrum."""
    system = _three_variable_var()
    innovations = var_to_innovations_state_space(system)
    frequencies = _grid(257)

    var_spectrum = cross_spectral_density(system, frequencies)
    iss_spectrum = innovations_spectral_density(innovations, frequencies)

    torch.testing.assert_close(iss_spectrum, var_spectrum, rtol=1e-10, atol=1e-11)


def test_f02_marginal_reduction_matches_scipy_generalized_dare_oracle():
    """F02: marginal innovations agree with an independent SciPy DARE backend."""
    system = _four_variable_iss()
    indices = (0, 2, 3)
    reduced = reduce_innovations_state_space(system, indices)

    a = system.transition
    c = system.observation
    k = system.gain
    v = system.innovation_covariance
    selector = torch.eye(c.shape[-2], dtype=c.dtype, device=c.device)[list(indices)]
    if c.ndim == 3:
        selector = selector.unsqueeze(0).expand(c.shape[0], -1, -1)
    reduced_c = selector @ c
    q = k @ v @ k.transpose(-1, -2)
    r = symmetrise(selector @ v @ selector.transpose(-1, -2))
    s = k @ v @ selector.transpose(-1, -2)
    p = solve_generalized_dare(a, reduced_c, q, r, s, backend="scipy")
    if p.ndim == 2:
        p = p.unsqueeze(0)
    expected_v = symmetrise(reduced_c @ p @ reduced_c.transpose(-1, -2) + r)
    numerator = a @ p @ reduced_c.transpose(-1, -2) + s
    identity = torch.eye(expected_v.shape[-1], dtype=expected_v.dtype).expand_as(expected_v)
    expected_k = numerator @ spd_solve(expected_v, identity)

    torch.testing.assert_close(reduced.observation, reduced_c, rtol=1e-9, atol=1e-11)
    torch.testing.assert_close(reduced.innovation_covariance, expected_v, rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(reduced.gain, expected_k, rtol=1e-8, atol=1e-10)


def test_f04_iid_three_variable_oir_reduces_to_static_gaussian_o_information():
    """F04: the no-lag O-information rate equals static O-information/sample."""
    dtype = torch.float64
    coefficients = torch.zeros((1, 3, 3), dtype=dtype)
    covariance = torch.tensor(
        [[1.0, 0.25, 0.12], [0.25, 0.9, -0.08], [0.12, -0.08, 0.8]],
        dtype=dtype,
    )
    system = var_to_innovations_state_space(build_var_system(coefficients, covariance))

    # For three variables O-information equals co-information:
    # I(X0;X1)+I(X0;X2)-I(X0;X1,X2).
    rate = (
        gaussian_mutual_information_rate(system, [0], [1], base=2.0)
        + gaussian_mutual_information_rate(system, [0], [2], base=2.0)
        - gaussian_mutual_information_rate(system, [0], [1, 2], base=2.0)
    )
    static = o_information(covariance, base=2.0)
    torch.testing.assert_close(rate.squeeze(), static, rtol=1e-10, atol=1e-11)


def test_f06_three_source_pid_lattice_reconstructs_all_subset_temporal_mirs():
    """F06: the generic 18-node lattice conserves all source-subset MIRs."""
    system = _four_variable_iss()
    target = [3]
    frequencies = _grid(513)
    lattice = pid_lattice(3)
    assert len(lattice.antichains) == 18

    subset_rates = {}
    for mask in range(1, 8):
        sources = [index for index in range(3) if mask & (1 << index)]
        subset_rates[frozenset(sources)] = spectral_gaussian_mutual_information_rate(
            system, sources, target, frequencies, base=math.e
        )

    redundancy_f = _spectral_redundancy(lattice, subset_rates)
    atoms_f = pid_mobius_inversion(redundancy_f, lattice)
    atoms_t = integrate_spectral_rate(atoms_f.transpose(-1, -2), frequencies)
    reconstructed_t = pid_redundancy_from_atoms(atoms_t, lattice)

    for subset in subset_rates:
        sources = sorted(subset)
        direct = gaussian_mutual_information_rate(system, sources, target, base=math.e)
        node = lattice.index((sources,))
        torch.testing.assert_close(
            reconstructed_t[..., node], direct, rtol=2e-8, atol=2e-9
        )
    joint = gaussian_mutual_information_rate(system, [0, 1, 2], target, base=math.e)
    torch.testing.assert_close(atoms_t.sum(dim=-1), joint, rtol=2e-8, atol=2e-9)


def test_f07_pdgc_kernel_reconstructs_direct_temporal_state_space_gc():
    """F07: spectral GC lattice conserves temporal Barnett/Seth state-space GC."""
    system = var_to_innovations_state_space(_three_variable_var())
    target = [2]
    frequencies = _grid(1025)
    lattice = pid_lattice(2)

    subset_gc = {
        frozenset({0}): state_space_spectral_mvgc(
            system, [0], target, frequencies, conditional=(), base=math.e
        ),
        frozenset({1}): state_space_spectral_mvgc(
            system, [1], target, frequencies, conditional=(), base=math.e
        ),
        frozenset({0, 1}): state_space_spectral_mvgc(
            system, [0, 1], target, frequencies, conditional=(), base=math.e
        ),
    }
    redundancy_f = _spectral_redundancy(lattice, subset_gc)
    atoms_f = pid_mobius_inversion(redundancy_f, lattice)
    atoms_t = integrate_spectral_rate(atoms_f.transpose(-1, -2), frequencies)
    reconstructed_t = pid_redundancy_from_atoms(atoms_t, lattice)

    for subset in subset_gc:
        sources = sorted(subset)
        direct = state_space_temporal_mvgc(
            system, sources, target, conditional=(), base=math.e
        )
        node = lattice.index((sources,))
        torch.testing.assert_close(
            reconstructed_t[..., node], direct, rtol=3e-7, atol=3e-8
        )
    direct_joint = state_space_temporal_mvgc(
        system, [0, 1], target, conditional=(), base=math.e
    )
    torch.testing.assert_close(atoms_t.sum(dim=-1), direct_joint, rtol=3e-7, atol=3e-8)


def test_f09_varma_aoki_realization_matches_direct_varma_spectrum():
    """F09: nontrivial VARMA and its Aoki ISS realization have identical PSDs."""
    dtype = torch.float64
    ar = torch.tensor(
        [
            [[0.35, 0.06], [-0.04, 0.28]],
            [[0.07, 0.00], [0.02, -0.05]],
        ],
        dtype=dtype,
    )
    ma = torch.tensor(
        [
            [[0.20, 0.00], [0.00, 0.20]],
            [[0.08, 0.00], [0.00, 0.08]],
        ],
        dtype=dtype,
    )
    covariance = torch.tensor([[1.0, 0.15], [0.15, 0.8]], dtype=dtype)
    system = varma_to_innovations_state_space(ar, ma, covariance)
    frequencies = _grid(257)
    spectrum_iss = innovations_spectral_density(system, frequencies)

    complex_dtype = torch.complex128
    eye = torch.eye(2, dtype=complex_dtype)
    ar_c = ar.to(complex_dtype)
    ma_c = ma.to(complex_dtype)
    covariance_c = covariance.to(complex_dtype)
    direct = []
    for frequency in frequencies:
        z1 = torch.exp(-2j * torch.pi * frequency)
        a_poly = eye - ar_c[0] * z1 - ar_c[1] * z1**2
        b_poly = eye + ma_c[0] * z1 + ma_c[1] * z1**2
        transfer = torch.linalg.solve(a_poly, b_poly)
        direct.append(transfer @ covariance_c @ transfer.conj().transpose(-1, -2))
    spectrum_direct = torch.stack(direct)

    torch.testing.assert_close(spectrum_iss, spectrum_direct, rtol=2e-9, atol=2e-10)


def test_f11_downsampling_scales_two_and_five_match_independent_dare_backends():
    """F11: exact downsampling at scales 2/5 is stable across DARE backends."""
    from complextorch import downsample_innovations_state_space

    system = var_to_innovations_state_space(_three_variable_var())
    for factor in (2, 5):
        torch_result = downsample_innovations_state_space(
            system, factor, backend="torch"
        )
        scipy_result = downsample_innovations_state_space(
            system, factor, backend="scipy"
        )
        expected_transition = torch.linalg.matrix_power(system.transition, factor)
        torch.testing.assert_close(
            torch_result.transition, expected_transition, rtol=1e-10, atol=1e-11
        )
        torch.testing.assert_close(
            torch_result.transition, scipy_result.transition, rtol=1e-10, atol=1e-11
        )
        torch.testing.assert_close(
            torch_result.innovation_covariance,
            scipy_result.innovation_covariance,
            rtol=2e-7,
            atol=2e-9,
        )
        torch.testing.assert_close(
            torch_result.gain, scipy_result.gain, rtol=2e-7, atol=2e-9
        )
