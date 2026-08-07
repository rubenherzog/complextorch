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


def _system():
    dtype = torch.float64
    coefficients = torch.tensor(
        [
            [[0.45, 0.12, 0.00], [0.05, 0.36, 0.08], [0.18, -0.10, 0.30]],
            [[0.08, 0.00, 0.00], [0.00, -0.06, 0.00], [0.04, 0.03, 0.05]],
        ],
        dtype=dtype,
    )
    covariance = torch.tensor(
        [[1.00, 0.22, 0.08], [0.22, 0.85, -0.05], [0.08, -0.05, 0.70]],
        dtype=dtype,
    )
    return var_to_innovations_state_space(build_var_system(coefficients, covariance))


def _grid(n=1025):
    return torch.linspace(0.0, 0.5, n, dtype=torch.float64)


def test_innovations_spectrum_is_hermitian_positive_definite():
    spectrum = innovations_spectral_density(_system(), _grid(65))
    torch.testing.assert_close(
        spectrum, spectrum.conj().transpose(-1, -2), rtol=1e-12, atol=1e-12
    )
    assert bool(torch.isfinite(hermitian_logdet(spectrum)).all().item())


def test_spectral_mir_integrates_to_direct_temporal_mir():
    system = _system()
    frequency = _grid()
    spectral = spectral_gaussian_mutual_information_rate(
        system, [0], [1, 2], frequency, base=math.e
    )
    temporal = gaussian_mutual_information_rate(system, [0], [1, 2], base=math.e)
    torch.testing.assert_close(
        integrate_spectral_rate(spectral, frequency), temporal, rtol=1e-8, atol=1e-9
    )


def test_spectral_te_integrates_to_direct_temporal_te():
    system = _system()
    frequency = _grid()
    spectral = spectral_gaussian_transfer_entropy_rate(
        system, [0, 1], [2], frequency, base=math.e
    )
    temporal = gaussian_transfer_entropy_rate(system, [0, 1], [2], base=math.e)
    torch.testing.assert_close(
        integrate_spectral_rate(spectral, frequency), temporal, rtol=1e-8, atol=1e-9
    )


def test_faes_mir_decomposition_matches_direct_temporal_components():
    system = _system()
    frequency = _grid()
    mir = spectral_gaussian_mutual_information_rate(system, [0], [2], frequency)
    te_0_2 = spectral_gaussian_transfer_entropy_rate(system, [0], [2], frequency)
    te_2_0 = spectral_gaussian_transfer_entropy_rate(system, [2], [0], frequency)
    instant = mir - te_0_2 - te_2_0

    torch.testing.assert_close(
        integrate_spectral_rate(mir, frequency),
        gaussian_mutual_information_rate(system, [0], [2]),
        rtol=1e-8,
        atol=1e-9,
    )
    torch.testing.assert_close(
        integrate_spectral_rate(te_0_2, frequency),
        gaussian_transfer_entropy_rate(system, [0], [2]),
        rtol=1e-8,
        atol=1e-9,
    )
    torch.testing.assert_close(
        integrate_spectral_rate(te_2_0, frequency),
        gaussian_transfer_entropy_rate(system, [2], [0]),
        rtol=1e-8,
        atol=1e-9,
    )
    torch.testing.assert_close(
        integrate_spectral_rate(instant, frequency),
        gaussian_instantaneous_information_rate(system, [0], [2]),
        rtol=1e-8,
        atol=1e-9,
    )


def test_pid_lattice_matches_two_source_formulas_and_three_source_size():
    lattice = pid_lattice(2)
    assert len(lattice.antichains) == 4
    assert len(pid_lattice(3).antichains) == 18

    redundancy = torch.zeros(4, dtype=torch.float64)
    redundancy[lattice.index(([0],))] = 0.7
    redundancy[lattice.index(([1],))] = 0.5
    redundancy[lattice.index(([0, 1],))] = 1.1
    redundancy[lattice.index(([0], [1]))] = 0.3
    atoms = pid_mobius_inversion(redundancy, lattice)

    torch.testing.assert_close(atoms[lattice.index(([0], [1]))], torch.tensor(0.3))
    torch.testing.assert_close(atoms[lattice.index(([0],))], torch.tensor(0.4))
    torch.testing.assert_close(atoms[lattice.index(([1],))], torch.tensor(0.2))
    torch.testing.assert_close(atoms[lattice.index(([0, 1],))], torch.tensor(0.2))
    torch.testing.assert_close(pid_redundancy_from_atoms(atoms, lattice), redundancy)


def _redundancy(lattice, subset_rates):
    values = []
    for antichain in lattice.antichains:
        components = torch.stack(
            [subset_rates[frozenset(subset)] for subset in antichain], dim=-1
        )
        values.append(components.min(dim=-1).values)
    return torch.stack(values, dim=-1)


def _integrated_atoms(lattice, subset_rates, frequency):
    atoms_f = pid_mobius_inversion(_redundancy(lattice, subset_rates), lattice)
    return integrate_spectral_rate(atoms_f.transpose(-1, -2), frequency)


def test_pird_lattice_reconstructs_direct_temporal_mirs():
    system = _system()
    frequency = _grid()
    target = [2]
    lattice = pid_lattice(2)
    rates = {
        frozenset({0}): spectral_gaussian_mutual_information_rate(system, [0], target, frequency),
        frozenset({1}): spectral_gaussian_mutual_information_rate(system, [1], target, frequency),
        frozenset({0, 1}): spectral_gaussian_mutual_information_rate(system, [0, 1], target, frequency),
    }
    atoms = _integrated_atoms(lattice, rates, frequency)
    reconstructed = pid_redundancy_from_atoms(atoms, lattice)

    direct0 = gaussian_mutual_information_rate(system, [0], target)
    direct1 = gaussian_mutual_information_rate(system, [1], target)
    direct_joint = gaussian_mutual_information_rate(system, [0, 1], target)
    torch.testing.assert_close(reconstructed[..., lattice.index(([0],))], direct0, rtol=1e-8, atol=1e-9)
    torch.testing.assert_close(reconstructed[..., lattice.index(([1],))], direct1, rtol=1e-8, atol=1e-9)
    torch.testing.assert_close(reconstructed[..., lattice.index(([0, 1],))], direct_joint, rtol=1e-8, atol=1e-9)
    torch.testing.assert_close(atoms.sum(dim=-1), direct_joint, rtol=1e-8, atol=1e-9)


def test_te_lattice_reconstructs_direct_temporal_tes():
    system = _system()
    frequency = _grid()
    target = [2]
    lattice = pid_lattice(2)
    rates = {
        frozenset({0}): spectral_gaussian_transfer_entropy_rate(system, [0], target, frequency),
        frozenset({1}): spectral_gaussian_transfer_entropy_rate(system, [1], target, frequency),
        frozenset({0, 1}): spectral_gaussian_transfer_entropy_rate(system, [0, 1], target, frequency),
    }
    atoms = _integrated_atoms(lattice, rates, frequency)
    reconstructed = pid_redundancy_from_atoms(atoms, lattice)

    direct0 = gaussian_transfer_entropy_rate(system, [0], target)
    direct1 = gaussian_transfer_entropy_rate(system, [1], target)
    direct_joint = gaussian_transfer_entropy_rate(system, [0, 1], target)
    torch.testing.assert_close(reconstructed[..., lattice.index(([0],))], direct0, rtol=1e-8, atol=1e-9)
    torch.testing.assert_close(reconstructed[..., lattice.index(([1],))], direct1, rtol=1e-8, atol=1e-9)
    torch.testing.assert_close(reconstructed[..., lattice.index(([0, 1],))], direct_joint, rtol=1e-8, atol=1e-9)
    torch.testing.assert_close(atoms.sum(dim=-1), direct_joint, rtol=1e-8, atol=1e-9)
