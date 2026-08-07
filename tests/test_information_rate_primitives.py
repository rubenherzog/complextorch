import math

import torch

from complextorch import build_var_system, var_to_innovations_state_space
from complextorch.measures._pid_lattice import (
    pid_lattice,
    pid_mobius_inversion,
    pid_redundancy_from_atoms,
)
from complextorch.measures.rates import (
    gaussian_instantaneous_information_rate,
    gaussian_mutual_information_rate,
    gaussian_transfer_entropy_rate,
    spectral_gaussian_mutual_information_rate,
    spectral_gaussian_transfer_entropy_rate,
)
from complextorch.spectra import (
    hermitian_logdet,
    innovations_spectral_density,
    integrate_spectral_rate,
)


def _three_variable_system():
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
    return var_to_innovations_state_space(build_var_system(coefficients, covariance))


def _hop_grid(nfft: int, *, dtype=torch.float64):
    return torch.arange(nfft, dtype=dtype) / (2.0 * nfft)


def test_innovations_spectrum_is_hermitian_positive_definite():
    system = _three_variable_system()
    frequencies = _hop_grid(64)
    spectrum = innovations_spectral_density(system, frequencies)

    torch.testing.assert_close(
        spectrum, spectrum.conj().transpose(-1, -2), rtol=1e-12, atol=1e-12
    )
    logdet = hermitian_logdet(spectrum)
    assert bool(torch.isfinite(logdet).all().item())


def test_spectral_mir_integrates_to_direct_temporal_mir():
    system = _three_variable_system()
    frequencies = _hop_grid(4096)

    spectral = spectral_gaussian_mutual_information_rate(
        system, [0], [1, 2], frequencies, base=math.e
    )
    integrated = integrate_spectral_rate(
        spectral, frequencies, half_open=True
    )
    temporal = gaussian_mutual_information_rate(
        system, [0], [1, 2], base=math.e
    )

    torch.testing.assert_close(integrated, temporal, rtol=2e-5, atol=2e-6)


def test_spectral_te_integrates_to_direct_temporal_te_with_correlated_innovations():
    system = _three_variable_system()
    frequencies = _hop_grid(4096)

    spectral = spectral_gaussian_transfer_entropy_rate(
        system, [0, 1], [2], frequencies, base=math.e
    )
    integrated = integrate_spectral_rate(
        spectral, frequencies, half_open=True
    )
    temporal = gaussian_transfer_entropy_rate(
        system, [0, 1], [2], base=math.e
    )

    torch.testing.assert_close(integrated, temporal, rtol=2e-5, atol=2e-6)


def test_faes_mir_decomposition_matches_three_direct_temporal_terms():
    system = _three_variable_system()
    frequencies = _hop_grid(4096)

    mir_spectral = spectral_gaussian_mutual_information_rate(
        system, [0], [2], frequencies
    )
    te_0_to_2_spectral = spectral_gaussian_transfer_entropy_rate(
        system, [0], [2], frequencies
    )
    te_2_to_0_spectral = spectral_gaussian_transfer_entropy_rate(
        system, [2], [0], frequencies
    )
    instantaneous_spectral = (
        mir_spectral - te_0_to_2_spectral - te_2_to_0_spectral
    )

    mir_frequency = integrate_spectral_rate(
        mir_spectral, frequencies, half_open=True
    )
    te_0_to_2_frequency = integrate_spectral_rate(
        te_0_to_2_spectral, frequencies, half_open=True
    )
    te_2_to_0_frequency = integrate_spectral_rate(
        te_2_to_0_spectral, frequencies, half_open=True
    )
    instantaneous_frequency = integrate_spectral_rate(
        instantaneous_spectral, frequencies, half_open=True
    )

    mir_temporal = gaussian_mutual_information_rate(system, [0], [2])
    te_0_to_2_temporal = gaussian_transfer_entropy_rate(system, [0], [2])
    te_2_to_0_temporal = gaussian_transfer_entropy_rate(system, [2], [0])
    instantaneous_temporal = gaussian_instantaneous_information_rate(
        system, [0], [2]
    )

    torch.testing.assert_close(mir_frequency, mir_temporal, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(
        te_0_to_2_frequency, te_0_to_2_temporal, rtol=2e-5, atol=2e-6
    )
    torch.testing.assert_close(
        te_2_to_0_frequency, te_2_to_0_temporal, rtol=2e-5, atol=2e-6
    )
    torch.testing.assert_close(
        instantaneous_frequency,
        instantaneous_temporal,
        rtol=3e-5,
        atol=3e-6,
    )
    torch.testing.assert_close(
        mir_temporal,
        te_0_to_2_temporal + te_2_to_0_temporal + instantaneous_temporal,
        rtol=1e-9,
        atol=1e-10,
    )


def test_pid_lattice_has_expected_two_and_three_source_sizes_and_two_source_atoms():
    lattice2 = pid_lattice(2)
    lattice3 = pid_lattice(3)
    assert len(lattice2.antichains) == 4
    assert len(lattice3.antichains) == 18

    redundancy = torch.zeros(4, dtype=torch.float64)
    redundancy[lattice2.index(([0],))] = 0.7
    redundancy[lattice2.index(([1],))] = 0.5
    redundancy[lattice2.index(([0, 1],))] = 1.1
    redundancy[lattice2.index(([0], [1]))] = 0.3
    atoms = pid_mobius_inversion(redundancy, lattice2)

    torch.testing.assert_close(
        atoms[lattice2.index(([0], [1]))], torch.tensor(0.3, dtype=atoms.dtype)
    )
    torch.testing.assert_close(
        atoms[lattice2.index(([0],))], torch.tensor(0.4, dtype=atoms.dtype)
    )
    torch.testing.assert_close(
        atoms[lattice2.index(([1],))], torch.tensor(0.2, dtype=atoms.dtype)
    )
    torch.testing.assert_close(
        atoms[lattice2.index(([0, 1],))], torch.tensor(0.2, dtype=atoms.dtype)
    )
    torch.testing.assert_close(
        pid_redundancy_from_atoms(atoms, lattice2), redundancy
    )


def _spectral_redundancy_from_subset_rates(lattice, subset_rates):
    values = []
    for antichain in lattice.antichains:
        components = torch.stack(
            [subset_rates[frozenset(subset)] for subset in antichain], dim=-1
        )
        values.append(components.min(dim=-1).values)
    return torch.stack(values, dim=-1)


def test_spectral_pird_mobius_reconstructs_direct_temporal_mirs():
    system = _three_variable_system()
    frequencies = _hop_grid(4096)
    target = [2]
    lattice = pid_lattice(2)

    subset_mir = {
        frozenset({0}): spectral_gaussian_mutual_information_rate(
            system, [0], target, frequencies
        ),
        frozenset({1}): spectral_gaussian_mutual_information_rate(
            system, [1], target, frequencies
        ),
        frozenset({0, 1}): spectral_gaussian_mutual_information_rate(
            system, [0, 1], target, frequencies
        ),
    }
    redundancy_frequency = _spectral_redundancy_from_subset_rates(
        lattice, subset_mir
    )
    atoms_frequency = pid_mobius_inversion(redundancy_frequency, lattice)
    atoms_temporal = integrate_spectral_rate(
        atoms_frequency.transpose(-1, -2), frequencies, half_open=True
    )
    reconstructed_temporal = pid_redundancy_from_atoms(atoms_temporal, lattice)

    direct_source0 = gaussian_mutual_information_rate(system, [0], target)
    direct_source1 = gaussian_mutual_information_rate(system, [1], target)
    direct_joint = gaussian_mutual_information_rate(system, [0, 1], target)

    torch.testing.assert_close(
        reconstructed_temporal[..., lattice.index(([0],))],
        direct_source0,
        rtol=2e-5,
        atol=2e-6,
    )
    torch.testing.assert_close(
        reconstructed_temporal[..., lattice.index(([1],))],
        direct_source1,
        rtol=2e-5,
        atol=2e-6,
    )
    torch.testing.assert_close(
        reconstructed_temporal[..., lattice.index(([0, 1],))],
        direct_joint,
        rtol=2e-5,
        atol=2e-6,
    )
    torch.testing.assert_close(
        atoms_temporal.sum(dim=-1), direct_joint, rtol=2e-5, atol=2e-6
    )


def test_spectral_te_lattice_reconstructs_direct_temporal_tes():
    system = _three_variable_system()
    frequencies = _hop_grid(4096)
    target = [2]
    lattice = pid_lattice(2)

    subset_te = {
        frozenset({0}): spectral_gaussian_transfer_entropy_rate(
            system, [0], target, frequencies
        ),
        frozenset({1}): spectral_gaussian_transfer_entropy_rate(
            system, [1], target, frequencies
        ),
        frozenset({0, 1}): spectral_gaussian_transfer_entropy_rate(
            system, [0, 1], target, frequencies
        ),
    }
    redundancy_frequency = _spectral_redundancy_from_subset_rates(
        lattice, subset_te
    )
    atoms_frequency = pid_mobius_inversion(redundancy_frequency, lattice)
    atoms_temporal = integrate_spectral_rate(
        atoms_frequency.transpose(-1, -2), frequencies, half_open=True
    )
    reconstructed_temporal = pid_redundancy_from_atoms(atoms_temporal, lattice)

    direct_source0 = gaussian_transfer_entropy_rate(system, [0], target)
    direct_source1 = gaussian_transfer_entropy_rate(system, [1], target)
    direct_joint = gaussian_transfer_entropy_rate(system, [0, 1], target)

    torch.testing.assert_close(
        reconstructed_temporal[..., lattice.index(([0],))],
        direct_source0,
        rtol=2e-5,
        atol=2e-6,
    )
    torch.testing.assert_close(
        reconstructed_temporal[..., lattice.index(([1],))],
        direct_source1,
        rtol=2e-5,
        atol=2e-6,
    )
    torch.testing.assert_close(
        reconstructed_temporal[..., lattice.index(([0, 1],))],
        direct_joint,
        rtol=2e-5,
        atol=2e-6,
    )
    torch.testing.assert_close(
        atoms_temporal.sum(dim=-1), direct_joint, rtol=2e-5, atol=2e-6
    )
