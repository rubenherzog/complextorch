import math

import pytest
import torch

from complextorch import (
    InnovationsStateSpace,
    build_var_system,
    downsample_innovations_state_space,
    gaussian_instantaneous_information_rate,
    gaussian_mutual_information_rate,
    gaussian_transfer_entropy_rate,
    innovations_spectral_density,
    integrate_spectral_rate,
    spectral_gaussian_mutual_information_rate,
    spectral_gaussian_transfer_entropy_rate,
    var_to_innovations_state_space,
)
from complextorch.measures._pid_lattice import (
    pid_lattice,
    pid_mobius_inversion,
    pid_redundancy_from_atoms,
)
from complextorch.measures.mvgc import (
    state_space_spectral_mvgc,
    state_space_temporal_mvgc,
)


def _as_scalar(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(-1)[0]


def _grid(dtype=torch.float64, n=2049):
    return torch.linspace(0.0, 0.5, n, dtype=dtype)


def _iss(coefficients, covariance, *, dtype=torch.float64):
    coefficients = torch.as_tensor(coefficients, dtype=dtype)
    covariance = torch.as_tensor(covariance, dtype=dtype)
    return var_to_innovations_state_space(build_var_system(coefficients, covariance))


def _iid_independent(dtype=torch.float64):
    return _iss(
        torch.zeros((1, 2, 2), dtype=dtype),
        torch.eye(2, dtype=dtype),
        dtype=dtype,
    )


def _iid_correlated(dtype=torch.float64):
    return _iss(
        torch.zeros((1, 2, 2), dtype=dtype),
        torch.tensor([[1.0, 0.45], [0.45, 1.4]], dtype=dtype),
        dtype=dtype,
    )


def _unidirectional(dtype=torch.float64):
    # X0 drives X1 and X1 has no lagged path back to X0.
    coefficients = torch.tensor(
        [[[0.55, 0.00], [0.42, 0.35]]], dtype=dtype
    )
    covariance = torch.tensor([[1.0, 0.0], [0.0, 0.8]], dtype=dtype)
    return _iss(coefficients, covariance, dtype=dtype)


def _three_source_target(dtype=torch.float64):
    coefficients = torch.tensor(
        [
            [
                [0.35, 0.00, 0.00, 0.00],
                [0.00, 0.32, 0.00, 0.00],
                [0.00, 0.00, 0.28, 0.00],
                [0.28, 0.18, -0.15, 0.30],
            ]
        ],
        dtype=dtype,
    )
    covariance = torch.tensor(
        [
            [1.00, 0.18, 0.02, 0.03],
            [0.18, 0.90, -0.08, 0.01],
            [0.02, -0.08, 0.75, -0.02],
            [0.03, 0.01, -0.02, 0.85],
        ],
        dtype=dtype,
    )
    return _iss(coefficients, covariance, dtype=dtype)


def _block_transform(system: InnovationsStateSpace, transform: torch.Tensor):
    """Apply an invertible instantaneous observation transform exactly."""
    transform = transform.to(
        dtype=system.transition.dtype, device=system.transition.device
    )
    inverse_right = torch.linalg.solve(
        transform.transpose(-1, -2), system.gain.transpose(-1, -2)
    ).transpose(-1, -2)
    return InnovationsStateSpace(
        transition=system.transition,
        observation=transform @ system.observation,
        gain=inverse_right,
        innovation_covariance=(
            transform
            @ system.innovation_covariance
            @ transform.transpose(-1, -2)
        ),
    )


def _simulate_var1(
    coefficient: torch.Tensor,
    covariance: torch.Tensor,
    n_samples: int,
    *,
    seed: int,
    burnin: int = 1000,
) -> torch.Tensor:
    """Independent VAR(1) simulator used only as a Monte-Carlo oracle."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    chol = torch.linalg.cholesky(covariance)
    total = n_samples + burnin
    data = torch.zeros((total, coefficient.shape[0]), dtype=coefficient.dtype)
    noise = torch.randn(
        (total, coefficient.shape[0]), dtype=coefficient.dtype, generator=generator
    ) @ chol.transpose(-1, -2)
    for t in range(1, total):
        data[t] = coefficient @ data[t - 1] + noise[t]
    return data[burnin:]


def _fit_var1_ols(data: torch.Tensor):
    """Independent OLS VAR(1) fit; does not use ComplexTorch VAR fitting."""
    predictors = data[:-1]
    response = data[1:]
    solution = torch.linalg.lstsq(predictors, response).solution
    coefficient = solution.transpose(-1, -2)
    residuals = response - predictors @ solution
    covariance = residuals.transpose(-1, -2) @ residuals / residuals.shape[0]
    return coefficient, covariance


def _reference_leq(alpha, beta):
    """Independent Williams--Beer order written directly from its definition."""
    return all(
        any(set(left).issubset(set(right)) for left in alpha)
        for right in beta
    )


def _recursive_mobius_reference(redundancy, lattice):
    """Independent recursive Möbius inversion, without zeta/mobius matrices."""
    atoms = []
    for index, node in enumerate(lattice.antichains):
        lower_sum = torch.zeros_like(redundancy[..., index])
        for lower_index, lower in enumerate(lattice.antichains[:index]):
            if _reference_leq(lower, node) and not _reference_leq(node, lower):
                lower_sum = lower_sum + atoms[lower_index]
        atoms.append(redundancy[..., index] - lower_sum)
    return torch.stack(atoms, dim=-1)


def test_iid_independent_has_zero_information_and_predictive_rates():
    system = _iid_independent()
    frequency = _grid()

    mir = gaussian_mutual_information_rate(system, [0], [1])
    te01 = gaussian_transfer_entropy_rate(system, [0], [1])
    te10 = gaussian_transfer_entropy_rate(system, [1], [0])
    instantaneous = gaussian_instantaneous_information_rate(system, [0], [1])

    for value in (mir, te01, te10, instantaneous):
        torch.testing.assert_close(value, torch.zeros_like(value), atol=1e-11, rtol=0.0)

    spectral_mir = spectral_gaussian_mutual_information_rate(
        system, [0], [1], frequency
    )
    spectral_te = spectral_gaussian_transfer_entropy_rate(
        system, [0], [1], frequency
    )
    torch.testing.assert_close(
        spectral_mir, torch.zeros_like(spectral_mir), atol=1e-10, rtol=0.0
    )
    torch.testing.assert_close(
        spectral_te, torch.zeros_like(spectral_te), atol=1e-10, rtol=0.0
    )


def test_iid_correlated_is_purely_instantaneous_without_transfer_entropy():
    system = _iid_correlated()
    covariance = system.innovation_covariance
    if covariance.ndim == 3:
        covariance = covariance[0]
    expected = 0.5 * torch.log(
        covariance[0, 0] * covariance[1, 1] / torch.linalg.det(covariance)
    )

    mir = _as_scalar(gaussian_mutual_information_rate(system, [0], [1]))
    instantaneous = _as_scalar(
        gaussian_instantaneous_information_rate(system, [0], [1])
    )
    te01 = _as_scalar(gaussian_transfer_entropy_rate(system, [0], [1]))
    te10 = _as_scalar(gaussian_transfer_entropy_rate(system, [1], [0]))

    torch.testing.assert_close(mir, expected, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(instantaneous, expected, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(te01, torch.zeros_like(te01), atol=1e-11, rtol=0.0)
    torch.testing.assert_close(te10, torch.zeros_like(te10), atol=1e-11, rtol=0.0)


def test_unidirectional_var_has_forward_but_not_reverse_transfer_entropy():
    system = _unidirectional()
    frequency = _grid()

    forward = _as_scalar(gaussian_transfer_entropy_rate(system, [0], [1]))
    reverse = _as_scalar(gaussian_transfer_entropy_rate(system, [1], [0]))
    forward_gc = _as_scalar(
        state_space_temporal_mvgc(system, [0], [1], conditional=())
    )
    reverse_gc = _as_scalar(
        state_space_temporal_mvgc(system, [1], [0], conditional=())
    )

    assert forward > 1e-3
    torch.testing.assert_close(reverse, torch.zeros_like(reverse), atol=2e-10, rtol=0.0)
    torch.testing.assert_close(reverse_gc, torch.zeros_like(reverse_gc), atol=4e-10, rtol=0.0)
    torch.testing.assert_close(2.0 * forward, forward_gc, rtol=1e-9, atol=1e-10)

    forward_spectrum = spectral_gaussian_transfer_entropy_rate(
        system, [0], [1], frequency
    )
    forward_gc_spectrum = state_space_spectral_mvgc(
        system, [0], [1], frequency, conditional=()
    )
    torch.testing.assert_close(
        integrate_spectral_rate(forward_spectrum, frequency),
        forward,
        rtol=2e-7,
        atol=2e-9,
    )
    torch.testing.assert_close(
        integrate_spectral_rate(forward_gc_spectrum, frequency),
        forward_gc,
        rtol=2e-7,
        atol=2e-9,
    )


def test_independent_dynamic_blocks_have_zero_cross_block_information_rate():
    coefficients = torch.tensor(
        [[[0.60, 0.00], [0.00, -0.35]]], dtype=torch.float64
    )
    covariance = torch.diag(torch.tensor([1.2, 0.7], dtype=torch.float64))
    system = _iss(coefficients, covariance)

    mir = gaussian_mutual_information_rate(system, [0], [1])
    te01 = gaussian_transfer_entropy_rate(system, [0], [1])
    te10 = gaussian_transfer_entropy_rate(system, [1], [0])
    for value in (mir, te01, te10):
        torch.testing.assert_close(value, torch.zeros_like(value), atol=2e-10, rtol=0.0)


def test_integrated_innovations_spectrum_recovers_zero_lag_covariance():
    system = _unidirectional()
    frequency = _grid(n=8193)
    spectrum = innovations_spectral_density(system, frequency)
    integrated = 2.0 * torch.trapezoid(spectrum, frequency, dim=-3).real

    # Independent state covariance: solve vec(P)=(I-A⊗A)^-1 vec(K V K').
    transition = system.transition
    observation = system.observation
    gain = system.gain
    innovation = system.innovation_covariance
    if transition.ndim == 3:
        transition = transition[0]
        observation = observation[0]
        gain = gain[0]
        innovation = innovation[0]
    q = gain @ innovation @ gain.transpose(-1, -2)
    r = transition.shape[0]
    kron = torch.kron(transition, transition)
    vec_q = q.reshape(-1)
    vec_p = torch.linalg.solve(
        torch.eye(r * r, dtype=transition.dtype) - kron, vec_q
    )
    p = vec_p.reshape(r, r)
    expected = observation @ p @ observation.transpose(-1, -2) + innovation

    if integrated.ndim == 3:
        integrated = integrated[0]
    torch.testing.assert_close(integrated, expected, rtol=3e-6, atol=3e-7)


def test_information_rates_are_invariant_to_invertible_within_block_transforms():
    system = _iss(
        torch.tensor(
            [
                [
                    [0.42, 0.06, 0.00, 0.00],
                    [0.02, 0.38, 0.00, 0.00],
                    [0.15, -0.08, 0.31, 0.05],
                    [0.04, 0.12, -0.02, 0.28],
                ]
            ],
            dtype=torch.float64,
        ),
        torch.tensor(
            [
                [1.0, 0.1, 0.0, 0.0],
                [0.1, 0.8, 0.0, 0.0],
                [0.0, 0.0, 1.1, -0.15],
                [0.0, 0.0, -0.15, 0.9],
            ],
            dtype=torch.float64,
        ),
    )
    transform = torch.tensor(
        [
            [1.3, 0.2, 0.0, 0.0],
            [-0.1, 0.9, 0.0, 0.0],
            [0.0, 0.0, 0.8, 0.3],
            [0.0, 0.0, -0.2, 1.4],
        ],
        dtype=torch.float64,
    )
    transformed = _block_transform(system, transform)

    mir = gaussian_mutual_information_rate(system, [0, 1], [2, 3])
    mir_transformed = gaussian_mutual_information_rate(
        transformed, [0, 1], [2, 3]
    )
    te = gaussian_transfer_entropy_rate(system, [0, 1], [2, 3])
    te_transformed = gaussian_transfer_entropy_rate(
        transformed, [0, 1], [2, 3]
    )
    torch.testing.assert_close(mir_transformed, mir, rtol=2e-9, atol=2e-10)
    torch.testing.assert_close(te_transformed, te, rtol=2e-9, atol=2e-10)


def test_pid_mobius_matches_independent_recursive_reference_for_three_sources():
    lattice = pid_lattice(3)
    generator = torch.Generator().manual_seed(1729)
    redundancy = torch.randn(
        (7, len(lattice.antichains)), dtype=torch.float64, generator=generator
    )

    matrix_atoms = pid_mobius_inversion(redundancy, lattice)
    recursive_atoms = _recursive_mobius_reference(redundancy, lattice)
    torch.testing.assert_close(matrix_atoms, recursive_atoms, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(
        pid_redundancy_from_atoms(matrix_atoms, lattice),
        redundancy,
        rtol=0.0,
        atol=1e-12,
    )


def test_three_source_spectral_pid_reconstructs_every_temporal_subset_mir():
    system = _three_source_target()
    frequency = _grid(n=2049)
    target = [3]
    lattice = pid_lattice(3)

    subset_rates = {}
    for mask in range(1, 8):
        subset = tuple(index for index in range(3) if mask & (1 << index))
        subset_rates[frozenset(subset)] = spectral_gaussian_mutual_information_rate(
            system, subset, target, frequency
        )

    redundancy = []
    for antichain in lattice.antichains:
        terms = torch.stack(
            [subset_rates[frozenset(subset)] for subset in antichain], dim=-1
        )
        redundancy.append(terms.min(dim=-1).values)
    redundancy = torch.stack(redundancy, dim=-1)
    atoms = pid_mobius_inversion(redundancy, lattice)
    temporal_atoms = integrate_spectral_rate(
        atoms.transpose(-1, -2), frequency
    )
    reconstructed = pid_redundancy_from_atoms(temporal_atoms, lattice)

    for mask in range(1, 8):
        subset = tuple(index for index in range(3) if mask & (1 << index))
        direct = gaussian_mutual_information_rate(system, subset, target)
        position = lattice.index((subset,))
        torch.testing.assert_close(
            reconstructed[..., position], direct, rtol=4e-7, atol=4e-9
        )
    joint = gaussian_mutual_information_rate(system, [0, 1, 2], target)
    torch.testing.assert_close(
        temporal_atoms.sum(dim=-1), joint, rtol=4e-7, atol=4e-9
    )


def test_monte_carlo_ols_estimates_converge_to_theoretical_mir_te_and_gc():
    dtype = torch.float64
    coefficient = torch.tensor([[0.55, 0.00], [0.35, 0.30]], dtype=dtype)
    covariance = torch.tensor([[1.0, 0.15], [0.15, 0.9]], dtype=dtype)
    true_system = _iss(coefficient.unsqueeze(0), covariance)
    true_mir = _as_scalar(gaussian_mutual_information_rate(true_system, [0], [1]))
    true_te = _as_scalar(gaussian_transfer_entropy_rate(true_system, [0], [1]))
    true_gc = _as_scalar(
        state_space_temporal_mvgc(true_system, [0], [1], conditional=())
    )

    errors = {}
    for n_samples in (2000, 50000):
        run_errors = []
        for seed in (7, 19, 41, 83):
            data = _simulate_var1(
                coefficient, covariance, n_samples, seed=seed
            )
            estimated_a, estimated_v = _fit_var1_ols(data)
            estimated_system = _iss(estimated_a.unsqueeze(0), estimated_v)
            estimated = torch.stack(
                [
                    _as_scalar(
                        gaussian_mutual_information_rate(
                            estimated_system, [0], [1]
                        )
                    ),
                    _as_scalar(
                        gaussian_transfer_entropy_rate(
                            estimated_system, [0], [1]
                        )
                    ),
                    _as_scalar(
                        state_space_temporal_mvgc(
                            estimated_system, [0], [1], conditional=()
                        )
                    ),
                ]
            )
            truth = torch.stack([true_mir, true_te, true_gc])
            run_errors.append(torch.abs(estimated - truth))
        errors[n_samples] = torch.stack(run_errors).mean(dim=0)

    assert bool(torch.all(errors[50000] < errors[2000]).item())
    assert errors[50000][0] < 6e-3
    assert errors[50000][1] < 5e-3
    assert errors[50000][2] < 1e-2


def test_batch_results_match_explicit_loop_for_rates_spectra_and_downsampling():
    coefficients = torch.tensor(
        [
            [[[0.45, 0.00], [0.25, 0.30]]],
            [[[0.35, -0.12], [0.18, 0.40]]],
        ],
        dtype=torch.float64,
    )
    covariance = torch.tensor(
        [
            [[1.0, 0.10], [0.10, 0.8]],
            [[0.9, -0.08], [-0.08, 1.1]],
        ],
        dtype=torch.float64,
    )
    batched = var_to_innovations_state_space(
        build_var_system(coefficients, covariance)
    )
    frequency = _grid(n=257)

    batch_mir = gaussian_mutual_information_rate(batched, [0], [1])
    batch_te = gaussian_transfer_entropy_rate(batched, [0], [1])
    batch_spectrum = innovations_spectral_density(batched, frequency)
    batch_ds = downsample_innovations_state_space(batched, 2)

    for index in range(2):
        single = InnovationsStateSpace(
            batched.transition[index],
            batched.observation[index],
            batched.gain[index],
            batched.innovation_covariance[index],
        )
        torch.testing.assert_close(
            batch_mir[index], gaussian_mutual_information_rate(single, [0], [1])
        )
        torch.testing.assert_close(
            batch_te[index], gaussian_transfer_entropy_rate(single, [0], [1])
        )
        torch.testing.assert_close(
            batch_spectrum[index], innovations_spectral_density(single, frequency)
        )
        single_ds = downsample_innovations_state_space(single, 2)
        torch.testing.assert_close(batch_ds.transition[index], single_ds.transition)
        torch.testing.assert_close(batch_ds.observation[index], single_ds.observation)
        torch.testing.assert_close(batch_ds.gain[index], single_ds.gain)
        torch.testing.assert_close(
            batch_ds.innovation_covariance[index], single_ds.innovation_covariance
        )


def test_float32_tracks_float64_for_information_rates_and_spectral_integrals():
    system64 = _unidirectional(torch.float64)
    system32 = _unidirectional(torch.float32)
    frequency64 = _grid(torch.float64, 1025)
    frequency32 = frequency64.to(torch.float32)

    quantities64 = torch.stack(
        [
            _as_scalar(gaussian_mutual_information_rate(system64, [0], [1])),
            _as_scalar(gaussian_transfer_entropy_rate(system64, [0], [1])),
            _as_scalar(
                integrate_spectral_rate(
                    spectral_gaussian_mutual_information_rate(
                        system64, [0], [1], frequency64
                    ),
                    frequency64,
                )
            ),
            _as_scalar(
                integrate_spectral_rate(
                    spectral_gaussian_transfer_entropy_rate(
                        system64, [0], [1], frequency64
                    ),
                    frequency64,
                )
            ),
        ]
    )
    quantities32 = torch.stack(
        [
            _as_scalar(gaussian_mutual_information_rate(system32, [0], [1])),
            _as_scalar(gaussian_transfer_entropy_rate(system32, [0], [1])),
            _as_scalar(
                integrate_spectral_rate(
                    spectral_gaussian_mutual_information_rate(
                        system32, [0], [1], frequency32
                    ),
                    frequency32,
                )
            ),
            _as_scalar(
                integrate_spectral_rate(
                    spectral_gaussian_transfer_entropy_rate(
                        system32, [0], [1], frequency32
                    ),
                    frequency32,
                )
            ),
        ]
    ).to(torch.float64)
    torch.testing.assert_close(quantities32, quantities64, rtol=2e-4, atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_matches_cpu_for_public_faes_primitives():
    system_cpu = _unidirectional(torch.float64)
    frequency_cpu = _grid(torch.float64, 513)
    system_cuda = InnovationsStateSpace(
        system_cpu.transition.cuda(),
        system_cpu.observation.cuda(),
        system_cpu.gain.cuda(),
        system_cpu.innovation_covariance.cuda(),
    )
    frequency_cuda = frequency_cpu.cuda()

    cpu_mir = gaussian_mutual_information_rate(system_cpu, [0], [1])
    gpu_mir = gaussian_mutual_information_rate(system_cuda, [0], [1]).cpu()
    cpu_te = gaussian_transfer_entropy_rate(system_cpu, [0], [1])
    gpu_te = gaussian_transfer_entropy_rate(system_cuda, [0], [1]).cpu()
    cpu_spectrum = innovations_spectral_density(system_cpu, frequency_cpu)
    gpu_spectrum = innovations_spectral_density(system_cuda, frequency_cuda).cpu()

    torch.testing.assert_close(gpu_mir, cpu_mir, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(gpu_te, cpu_te, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(gpu_spectrum, cpu_spectrum, rtol=1e-9, atol=1e-10)
