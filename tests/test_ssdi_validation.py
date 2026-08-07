"""Scientific validation tests for state-space dynamical dependence.

The deterministic generators and subspace metrics live in
``scripts.validate_ssdi`` so the executable study and pytest assertions share
exactly the same systems and geometry code.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from complextorch import (
    dynamical_dependence,
    innovations_proxy_sequence,
    innovations_transfer_function,
    optimise_dynamical_dependence,
    orthonormalise_projection,
    proxy_dynamical_dependence,
    proxy_dynamical_dependence_gradient,
    spectral_dynamical_dependence,
    spectral_dynamical_dependence_gradient,
    solve_generalized_dare,
)
from scripts.validate_ssdi import (
    COMPLEXBOX_COMMIT,
    case_library,
    innovations_system_from_mask,
    local_minimum_clusters,
    planted_modular_system,
    principal_angles_rows,
    random_initial_projections,
    subspace_distance,
    tnet5_mask,
    tnet9_mask,
)


def _direct_proxy(projection: torch.Tensor, sequence: torch.Tensor) -> torch.Tensor:
    total = torch.zeros((), dtype=projection.dtype)
    for qk in sequence:
        mq = projection @ qk
        total = total + torch.sum(mq * mq)
        reduced = mq @ projection.T
        total = total - torch.sum(reduced * reduced)
    return total


def test_mvgc2_tnet_masks_are_exact() -> None:
    assert torch.equal(
        tnet5_mask(),
        torch.tensor(
            [
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 0, 1, 0, 0],
                [1, 0, 0, 1, 1],
                [0, 0, 0, 1, 1],
            ],
            dtype=torch.float64,
        ),
    )
    assert torch.equal(
        tnet9_mask(),
        torch.tensor(
            [
                [1, 0, 1, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 0, 1],
            ],
            dtype=torch.float64,
        ),
    )


@pytest.mark.parametrize("name", list(case_library().keys()))
def test_graph_structured_systems_are_stable(name: str) -> None:
    system = case_library()[name]
    radius = torch.max(torch.abs(torch.linalg.eigvals(system.transition))).real
    assert float(radius) < 1.0
    assert torch.allclose(radius, torch.tensor(0.72, dtype=radius.dtype), atol=1e-10)
    n = system.observation.shape[0]
    assert system.transition.shape == (n, n)
    assert system.observation.shape == (n, n)
    assert system.gain.shape == (n, n)
    assert torch.equal(system.innovation_covariance, torch.eye(n, dtype=torch.float64))


def test_planted_modular_macrospace_is_dynamically_closed() -> None:
    system, planted = planted_modular_system()
    exact = dynamical_dependence(system, planted, base=math.e)
    assert bool(torch.isfinite(exact))
    assert abs(float(exact)) < 1e-9


def test_proxy_sequence_matches_direct_state_space_recurrence() -> None:
    system = innovations_system_from_mask(tnet5_mask(), seed=123)
    sequence = innovations_proxy_sequence(system, lags=8)
    power = torch.eye(system.transition.shape[0], dtype=torch.float64)
    expected = []
    for lag in range(8):
        if lag:
            power = power @ system.transition
        expected.append(system.observation @ power @ system.gain)
    expected_tensor = torch.stack(expected)
    assert torch.allclose(sequence, expected_tensor, rtol=1e-12, atol=1e-12)


def test_proxy_equation_matches_independent_loop() -> None:
    system = innovations_system_from_mask(tnet5_mask(), seed=124)
    sequence = innovations_proxy_sequence(system, lags=7)
    projection = random_initial_projections(
        1, 2, 5, seed=125, dtype=torch.float64, device=torch.device("cpu")
    )[0]
    observed = proxy_dynamical_dependence(projection, sequence)
    expected = _direct_proxy(projection, sequence)
    assert torch.allclose(observed, expected, rtol=1e-12, atol=1e-12)


def test_proxy_analytic_gradient_matches_central_difference_direction() -> None:
    system = innovations_system_from_mask(tnet5_mask(), seed=126)
    sequence = innovations_proxy_sequence(system, lags=6)
    projection = random_initial_projections(
        1, 2, 5, seed=127, dtype=torch.float64, device=torch.device("cpu")
    )[0]
    gradient, magnitude = proxy_dynamical_dependence_gradient(projection, sequence)
    assert float(magnitude) > 0.0
    tangent = gradient / magnitude
    epsilon = 1e-6
    plus = orthonormalise_projection(projection + epsilon * tangent)
    minus = orthonormalise_projection(projection - epsilon * tangent)
    fd = (
        proxy_dynamical_dependence(plus, sequence)
        - proxy_dynamical_dependence(minus, sequence)
    ) / (2.0 * epsilon)
    analytic = torch.sum(gradient * tangent)
    assert torch.allclose(fd, analytic, rtol=2e-5, atol=2e-7)


def test_principal_angles_are_basis_invariant() -> None:
    projection = random_initial_projections(
        1, 2, 5, seed=128, dtype=torch.float64, device=torch.device("cpu")
    )[0]
    rotation = torch.tensor(
        [[0.6, -0.8], [0.8, 0.6]], dtype=torch.float64
    )
    same_subspace = rotation @ projection
    angles = principal_angles_rows(projection, same_subspace)
    assert torch.allclose(angles, torch.zeros_like(angles), atol=2e-8)
    assert subspace_distance(projection, same_subspace) < 3e-8


def test_local_minimum_clustering_is_invariant_to_row_basis() -> None:
    p = random_initial_projections(
        2, 2, 5, seed=129, dtype=torch.float64, device=torch.device("cpu")
    )
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64)
    projections = torch.stack((p[0], rotation @ p[0], p[1]))
    clusters = sorted(map(sorted, local_minimum_clusters(projections, angle_tolerance=1e-6)))
    assert [0, 1] in clusters
    assert [2] in clusters


@pytest.mark.parametrize("optimizer", ["complexbox", "riemannian_armijo"])
def test_both_optimizers_are_finite_orthonormal_and_improve_proxy(optimizer: str) -> None:
    system = innovations_system_from_mask(tnet5_mask(), seed=130)
    initial = random_initial_projections(
        6, 2, 5, seed=131, dtype=torch.float64, device=torch.device("cpu")
    )
    sequence = innovations_proxy_sequence(system, lags=8)
    initial_objective = proxy_dynamical_dependence(initial, sequence)
    result = optimise_dynamical_dependence(
        system,
        initial,
        objective="proxy",
        optimizer=optimizer,
        lags=8,
        max_iterations=120,
        optimizer_options=(
            {"variant": 1, "initial_step_size": 1e-3}
            if optimizer == "complexbox"
            else {"initial_step_size": 1.0}
        ),
    )
    assert bool(torch.isfinite(result.objective).all())
    assert bool(torch.isfinite(result.projection).all())
    identity = torch.eye(2, dtype=torch.float64).expand(6, -1, -1)
    assert torch.allclose(
        result.projection @ result.projection.transpose(-1, -2),
        identity,
        rtol=1e-8,
        atol=1e-9,
    )
    assert float(result.objective.min()) <= float(initial_objective.min()) + 1e-10


@pytest.mark.parametrize("backend", ["scipy", "torch"])
def test_projected_generalized_dare_satisfies_state_space_equation(backend: str) -> None:
    system = innovations_system_from_mask(tnet5_mask(), seed=138)
    projection = random_initial_projections(
        1, 2, 5, seed=139, dtype=torch.float64, device=torch.device("cpu")
    )[0]
    a = system.transition
    c = projection @ system.observation
    v = system.innovation_covariance
    q = system.gain @ v @ system.gain.T
    r = projection @ v @ projection.T
    s = system.gain @ v @ projection.T
    p = solve_generalized_dare(a, c, q, r, s, backend=backend)

    innovation = c @ p @ c.T + r
    cross = a @ p @ c.T + s
    rhs = a @ p @ a.T + q - cross @ torch.linalg.solve(innovation, cross.T)
    assert torch.allclose(p, rhs, rtol=2e-8, atol=2e-10)

    sign_r, logdet_r = torch.linalg.slogdet(r)
    sign_v, logdet_v = torch.linalg.slogdet(innovation)
    assert float(sign_r) > 0.0 and float(sign_v) > 0.0
    direct_dd = logdet_v - logdet_r
    public_dd = dynamical_dependence(system, projection, base=math.e)
    assert torch.allclose(public_dd, direct_dd, rtol=2e-8, atol=2e-10)


def test_complexbox_proxy_and_exact_dd_parity_when_installed() -> None:
    pytest.importorskip(
        "complexbox",
        reason=f"external parity requires ComplexBox pinned at {COMPLEXBOX_COMMIT}",
    )
    from complexbox.ssdi.dd import cak2ddx, iss2cak, iss2dd

    system = innovations_system_from_mask(tnet5_mask(), seed=132)
    projection = random_initial_projections(
        1, 2, 5, seed=133, dtype=torch.float64, device=torch.device("cpu")
    )[0]
    sequence = innovations_proxy_sequence(system, lags=9)
    torch_proxy = proxy_dynamical_dependence(projection, sequence)

    a = system.transition.numpy()
    c = system.observation.numpy()
    k = system.gain.numpy()
    l = projection.T.numpy()
    cb_sequence = iss2cak(a, c, k, 9)
    cb_proxy = cak2ddx(l, cb_sequence)
    assert np.allclose(sequence.numpy().transpose(1, 2, 0), cb_sequence, rtol=1e-12, atol=1e-12)
    assert np.allclose(float(torch_proxy), cb_proxy, rtol=1e-11, atol=1e-12)

    torch_exact = dynamical_dependence(system, projection, base=math.e)
    cb_exact = iss2dd(l, a, c, k)
    assert np.allclose(float(torch_exact), cb_exact, rtol=2e-8, atol=2e-10)


def test_complexbox_optimizer_endpoint_parity_when_installed() -> None:
    pytest.importorskip(
        "complexbox",
        reason=f"external parity requires ComplexBox pinned at {COMPLEXBOX_COMMIT}",
    )
    from complexbox.ssdi.dd import iss2cak
    from complexbox.ssdi.optimise import opt_gd_ddx_mruns

    system = innovations_system_from_mask(tnet5_mask(), seed=134)
    initial = random_initial_projections(
        5, 2, 5, seed=135, dtype=torch.float64, device=torch.device("cpu")
    )
    torch_result = optimise_dynamical_dependence(
        system,
        initial,
        objective="proxy",
        optimizer="complexbox",
        lags=8,
        max_iterations=100,
        optimizer_options={"variant": 1, "initial_step_size": 1e-3},
    )
    cak = iss2cak(
        system.transition.numpy(),
        system.observation.numpy(),
        system.gain.numpy(),
        8,
    )
    dds, ls, _, _ = opt_gd_ddx_mruns(
        cak,
        initial.numpy().transpose(2, 1, 0),
        maxiters=100,
        variant=1,
        gdsig0=1e-3,
    )
    cb_rows = torch.as_tensor(ls.transpose(2, 1, 0), dtype=torch.float64)
    assert np.allclose(torch_result.objective.numpy(), dds, rtol=2e-10, atol=2e-12)
    distances = torch.tensor(
        [
            subspace_distance(torch_result.projection[i], cb_rows[i])
            for i in range(len(dds))
        ]
    )
    assert float(distances.max()) < 2e-7


def test_complexbox_spectral_objective_and_gradient_parity_when_installed() -> None:
    pytest.importorskip(
        "complexbox",
        reason=f"external parity requires ComplexBox pinned at {COMPLEXBOX_COMMIT}",
    )
    from complexbox.ssdi.dd import trfun2dd, trfun2ddgrad

    system = innovations_system_from_mask(tnet5_mask(), seed=136)
    projection = random_initial_projections(
        1, 2, 5, seed=137, dtype=torch.float64, device=torch.device("cpu")
    )[0]
    frequencies = torch.linspace(0.0, 0.5, 65, dtype=torch.float64)
    transfer = innovations_transfer_function(system, frequencies)
    torch_value = spectral_dynamical_dependence(projection, transfer)
    torch_gradient, torch_magnitude = spectral_dynamical_dependence_gradient(
        projection, transfer
    )

    cb_transfer = transfer.numpy().transpose(1, 2, 0)
    cb_value, _ = trfun2dd(projection.T.numpy(), cb_transfer)
    cb_gradient, cb_magnitude = trfun2ddgrad(projection.T.numpy(), cb_transfer)
    assert np.allclose(float(torch_value), cb_value, rtol=3e-10, atol=3e-12)
    assert np.allclose(
        torch_gradient.numpy(),
        cb_gradient.T,
        rtol=2e-8,
        atol=2e-10,
    )
    assert np.allclose(float(torch_magnitude), cb_magnitude, rtol=2e-8, atol=2e-10)
