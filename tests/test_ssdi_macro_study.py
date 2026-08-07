"""Lightweight contracts for the collaborator-requested SSDI macro study.

The expensive experiment (100 random restarts x up to 10,000 iterations for
seven macro dimensions) is deliberately not a CI test.  These tests validate
the exact topology, storage/orientation, optimization-history contract, and the
geometry/loadings used by that later experiment.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.ssdi_macro_study import (
    coordinate_axis_distances,
    fixed_modular_8_mask,
    fixed_modular_8_system,
    grassmann_distance_matrix,
    macro_dimensions,
    micro_macro_loadings,
    run_macro_dimension_sweep,
)
from scripts.validate_ssdi import COMPLEXBOX_COMMIT, random_initial_projections


def test_fixed_modular_8_topology_is_exact() -> None:
    mask = fixed_modular_8_mask()
    assert mask.shape == (8, 8)
    assert torch.equal(mask[:2, :2], torch.ones((2, 2), dtype=mask.dtype))
    assert torch.equal(mask[2:, 2:], torch.ones((6, 6), dtype=mask.dtype))

    # State-matrix orientation is [target, source].  The only cross-module edge
    # is node 1 -> node 2; there is no 6-module -> 2-module feedback edge.
    assert mask[2, 1] == 1
    assert torch.count_nonzero(mask[2:, :2]) == 1
    assert torch.count_nonzero(mask[:2, 2:]) == 0


def test_fixed_modular_8_system_is_stable_and_has_all_macro_dimensions() -> None:
    system = fixed_modular_8_system()
    radius = torch.max(torch.abs(torch.linalg.eigvals(system.transition))).real
    assert torch.allclose(radius, torch.tensor(0.72, dtype=radius.dtype), atol=1e-10)
    assert system.transition.shape == (8, 8)
    assert system.observation.shape == (8, 8)
    assert system.gain.shape == (8, 8)
    assert system.innovation_covariance.shape == (8, 8)
    assert macro_dimensions(system) == (1, 2, 3, 4, 5, 6, 7)


def test_macro_sweep_preserves_row_projection_and_history_contract() -> None:
    system = fixed_modular_8_system(seed=211)
    restarts = 3
    results = run_macro_dimension_sweep(
        system,
        restarts=restarts,
        max_iterations=4,
        optimizer="complexbox",
        objective="proxy",
        lags=4,
        history=True,
        seed=212,
        optimizer_options={"variant": 1, "initial_step_size": 1e-3},
    )
    assert tuple(results) == (1, 2, 3, 4, 5, 6, 7)

    for m, result in results.items():
        assert result.projection.shape == (restarts, m, 8)
        assert result.objective.shape == (restarts,)
        assert result.history is not None
        assert result.history.shape[0] == restarts
        assert result.history.shape[-1] == 4
        assert torch.isfinite(result.history[..., 0]).any(dim=1).all()
        assert bool(torch.isfinite(result.objective).all())
        assert bool(torch.isfinite(result.projection).all())

        # Optimizer results are stored sorted by ascending final objective; the
        # first projection is therefore the minimum-DD candidate highlighted by
        # the later trajectory figure.
        if restarts > 1:
            assert bool(torch.all(torch.diff(result.objective) >= -1e-12))

        identity = torch.eye(m, dtype=result.projection.dtype).expand(restarts, -1, -1)
        gram = result.projection @ result.projection.transpose(-1, -2)
        assert torch.allclose(gram, identity, rtol=1e-8, atol=1e-9)


def test_riemannian_backend_uses_same_projection_storage_contract() -> None:
    system = fixed_modular_8_system(seed=213)
    # Use the public optimizer directly through the same sweep helper; only a
    # few iterations/restarts are needed to test backend-independent storage.
    results = run_macro_dimension_sweep(
        system,
        restarts=2,
        max_iterations=2,
        optimizer="riemannian_armijo",
        objective="proxy",
        lags=3,
        history=True,
        seed=214,
        optimizer_options={"initial_step_size": 1.0},
    )
    for m, result in results.items():
        assert result.projection.shape == (2, m, 8)
        assert result.history is not None
        assert result.history.shape[0] == 2
        assert result.history.shape[-1] == 4


def test_gmetrics_distance_matrix_has_100_run_target_shape() -> None:
    projections = random_initial_projections(
        100,
        2,
        8,
        seed=215,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    distances = grassmann_distance_matrix(projections)
    assert distances.shape == (100, 100)
    assert torch.equal(torch.diagonal(distances), torch.zeros(100, dtype=distances.dtype))
    assert torch.allclose(distances, distances.T, rtol=0.0, atol=1e-12)
    assert bool(torch.all((distances >= 0.0) & (distances <= 1.0 + 1e-12)))


def test_micro_macro_loadings_are_basis_invariant_and_sum_to_dimension() -> None:
    projection = random_initial_projections(
        1,
        3,
        8,
        seed=216,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )[0]
    rotation = torch.linalg.qr(
        torch.tensor(
            [[0.7, -0.4, 0.2], [0.3, 0.8, -0.5], [-0.6, 0.1, 0.7]],
            dtype=torch.float64,
        )
    ).Q
    rotated = rotation @ projection

    loading = micro_macro_loadings(projection)
    rotated_loading = micro_macro_loadings(rotated)
    assert loading.shape == (8,)
    assert torch.allclose(loading, rotated_loading, rtol=1e-11, atol=1e-12)
    assert torch.allclose(loading.sum(), torch.tensor(3.0, dtype=loading.dtype), atol=1e-12)
    assert bool(torch.all((loading >= 0.0) & (loading <= 1.0 + 1e-12)))

    distances = coordinate_axis_distances(projection)
    rotated_distances = coordinate_axis_distances(rotated)
    assert torch.allclose(distances, rotated_distances, rtol=1e-11, atol=1e-12)
    assert bool(torch.all((distances >= 0.0) & (distances <= 1.0 + 1e-12)))


def test_complexbox_gmetrics_and_loading_parity_when_installed() -> None:
    pytest.importorskip(
        "complexbox",
        reason=f"external parity requires ComplexBox pinned at {COMPLEXBOX_COMMIT}",
    )
    from complexbox.ssdi._grassmann import gmetrics, gmetrics1, habeta

    projections = random_initial_projections(
        7,
        3,
        8,
        seed=217,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    observed_matrix = grassmann_distance_matrix(projections)
    # ComplexBox stores the same subspaces as columns (n,m,runs); conversion is
    # only at this parity boundary, never in ComplexTorch study storage.
    complexbox_l = projections.numpy().transpose(2, 1, 0)
    expected_matrix = gmetrics(complexbox_l)
    assert np.allclose(observed_matrix.numpy(), expected_matrix, rtol=2e-8, atol=2e-10)

    observed_loading = micro_macro_loadings(projections[0])
    observed_axis_distance = coordinate_axis_distances(projections[0])
    expected_loading = habeta(complexbox_l[:, :, 0])
    expected_axis_distance = gmetrics1(complexbox_l[:, :, 0])
    assert np.allclose(observed_loading.numpy(), expected_loading, rtol=2e-10, atol=2e-12)
    assert np.allclose(
        observed_axis_distance.numpy(),
        expected_axis_distance,
        rtol=2e-9,
        atol=2e-11,
    )
