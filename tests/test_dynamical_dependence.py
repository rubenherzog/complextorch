"""Equation-level regression tests for dynamical dependence."""

import numpy as np
import pytest
import torch
from scipy.linalg import solve_discrete_are

from complextorch import (
    InnovationsStateSpace,
    StateSpaceModel,
    build_var_system,
    dynamical_dependence,
    stochastic_interaction,
)


def _iss_fixture(dtype=torch.float64) -> InnovationsStateSpace:
    """Return a stable nontrivial innovations-form fixture."""
    return InnovationsStateSpace(
        transition=torch.tensor([[0.45, 0.10], [0.00, 0.30]], dtype=dtype),
        observation=torch.tensor([[1.00, 0.20], [0.25, 0.80]], dtype=dtype),
        gain=torch.tensor([[0.35, 0.08], [0.05, 0.28]], dtype=dtype),
        innovation_covariance=torch.tensor(
            [[1.00, 0.20], [0.20, 0.70]], dtype=dtype
        ),
    )


def _reference_dd(system: InnovationsStateSpace, projection: torch.Tensor) -> float:
    """Independent SciPy implementation of Barnett--Seth Eqs. (22), (33)-(34)."""
    a = system.transition.detach().cpu().numpy()
    c = system.observation.detach().cpu().numpy()
    k = system.gain.detach().cpu().numpy()
    v = system.innovation_covariance.detach().cpu().numpy()
    l = projection.detach().cpu().numpy()

    q = k @ v @ k.T
    reduced_c = l @ c
    reduced_r = l @ v @ l.T
    cross_s = k @ v @ l.T
    p = solve_discrete_are(a.T, reduced_c.T, q, reduced_r, s=cross_s)
    reduced_v = reduced_c @ p @ reduced_c.T + reduced_r
    return float(
        np.linalg.slogdet(reduced_v)[1] - np.linalg.slogdet(reduced_r)[1]
    )


def _reference_ss_innovations_covariance(
    system: StateSpaceModel, indices: tuple[int, ...] | None = None
) -> np.ndarray:
    """Independent SciPy innovations covariance for a general state-space model."""
    a = system.transition.detach().cpu().numpy()
    c = system.observation.detach().cpu().numpy()
    q = system.process_covariance.detach().cpu().numpy()
    r = system.observation_covariance.detach().cpu().numpy()
    if indices is not None:
        idx = np.asarray(indices)
        c = c[idx]
        r = r[np.ix_(idx, idx)]
    p = solve_discrete_are(a.T, c.T, q, r)
    return c @ p @ c.T + r


def _batched_iss_fixture() -> InnovationsStateSpace:
    """Return two independent ISS models with the same dimensions."""
    first = _iss_fixture()
    second = InnovationsStateSpace(
        transition=torch.tensor([[0.35, -0.05], [0.08, 0.25]], dtype=torch.float64),
        observation=torch.tensor([[0.90, 0.10], [-0.15, 0.75]], dtype=torch.float64),
        gain=torch.tensor([[0.30, 0.04], [0.02, 0.22]], dtype=torch.float64),
        innovation_covariance=torch.tensor(
            [[0.80, -0.10], [-0.10, 1.10]], dtype=torch.float64
        ),
    )
    return InnovationsStateSpace(
        transition=torch.stack([first.transition, second.transition]),
        observation=torch.stack([first.observation, second.observation]),
        gain=torch.stack([first.gain, second.gain]),
        innovation_covariance=torch.stack(
            [first.innovation_covariance, second.innovation_covariance]
        ),
    )


def _unbatch_iss(system: InnovationsStateSpace, index: int) -> InnovationsStateSpace:
    """Select one independent model from a batched ISS fixture."""
    return InnovationsStateSpace(
        system.transition[index],
        system.observation[index],
        system.gain[index],
        system.innovation_covariance[index],
    )


def test_dynamical_dependence_matches_independent_dare_reference():
    """DD must equal the exact reduced-DARE log-determinant ratio."""
    system = _iss_fixture()
    projection = torch.tensor([[1.0, 0.35]], dtype=torch.float64)
    expected = _reference_dd(system, projection)
    actual = dynamical_dependence(system, projection, base=np.e)
    torch.testing.assert_close(
        actual,
        torch.tensor(expected, dtype=actual.dtype),
        rtol=1e-8,
        atol=1e-10,
    )


def test_dynamical_dependence_matches_complexbox_iss2ce_fixture():
    """Match ComplexBox iss2ce for an arbitrary non-identity innovation covariance."""
    # Frozen from bmilinkovic/complexbox@87b5e2c, ssdi.dd.iss2ce. ComplexBox
    # uses a column basis L (n,m); ComplexTorch uses the operational row
    # projection L.T (m,n). Both use natural logarithms for this comparison.
    system = _iss_fixture()
    projection = torch.tensor([[1.0, 0.35]], dtype=torch.float64)
    expected = torch.tensor(0.00284786356182265, dtype=torch.float64)
    actual = dynamical_dependence(system, projection, base=np.e)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)


def test_dynamical_dependence_identity_projection_is_zero():
    """An invertible full-dimensional macro variable retains the full past."""
    system = _iss_fixture()
    projection = torch.eye(2, dtype=torch.float64)
    actual = dynamical_dependence(system, projection)
    torch.testing.assert_close(
        actual, torch.zeros_like(actual), rtol=0.0, atol=1e-10
    )


def test_dynamical_dependence_requires_projection():
    """DD must not silently reinterpret a legacy no-projection call."""
    with pytest.raises(TypeError):
        dynamical_dependence(_iss_fixture())


def test_dynamical_dependence_is_invariant_to_macro_basis_change():
    """Nonsingular changes of macro coordinates must leave DD unchanged."""
    system = _iss_fixture()
    projection = torch.tensor([[1.0, 0.35]], dtype=torch.float64)
    transformed = 3.7 * projection
    torch.testing.assert_close(
        dynamical_dependence(system, projection),
        dynamical_dependence(system, transformed),
        rtol=1e-8,
        atol=1e-10,
    )


def test_dynamical_dependence_batches_projection_candidates():
    """Independent projection candidates share one model without being pooled."""
    system = _iss_fixture()
    projections = torch.tensor(
        [[[1.0, 0.35]], [[0.20, 1.0]]], dtype=torch.float64
    )
    batched = dynamical_dependence(system, projections)
    expected = torch.stack(
        [dynamical_dependence(system, projection) for projection in projections]
    )
    assert batched.shape == (2,)
    torch.testing.assert_close(batched, expected, rtol=1e-9, atol=1e-11)


def test_dynamical_dependence_batches_systems_with_shared_projection():
    """A shared projection broadcasts across independent microscopic systems."""
    systems = _batched_iss_fixture()
    projection = torch.tensor([[1.0, 0.35]], dtype=torch.float64)
    batched = dynamical_dependence(systems, projection)
    expected = torch.stack(
        [
            dynamical_dependence(_unbatch_iss(systems, index), projection)
            for index in range(2)
        ]
    )
    assert batched.shape == (2,)
    torch.testing.assert_close(batched, expected, rtol=1e-9, atol=1e-11)


def test_dynamical_dependence_batches_paired_systems_and_projections():
    """Corresponding system/projection pairs are evaluated independently."""
    systems = _batched_iss_fixture()
    projections = torch.tensor(
        [[[1.0, 0.35]], [[0.20, 1.0]]], dtype=torch.float64
    )
    batched = dynamical_dependence(systems, projections)
    expected = torch.stack(
        [
            dynamical_dependence(
                _unbatch_iss(systems, index), projections[index]
            )
            for index in range(2)
        ]
    )
    torch.testing.assert_close(batched, expected, rtol=1e-9, atol=1e-11)


def test_dynamical_dependence_rejects_incompatible_batches():
    """System and projection batches may only match or broadcast from one."""
    systems = _batched_iss_fixture()
    projections = torch.tensor(
        [[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 0.5]]], dtype=torch.float64
    )
    with pytest.raises(ValueError, match="incompatible batch"):
        dynamical_dependence(systems, projections)


def test_dynamical_dependence_preserves_float32_dtype():
    """Torch dtype is preserved through the generalized DARE path."""
    system = _iss_fixture(torch.float32)
    projection = torch.tensor([[1.0, 0.35]], dtype=torch.float32)
    value = dynamical_dependence(system, projection)
    assert value.dtype == torch.float32
    assert value.device == system.observation.device
    assert torch.isfinite(value)


def test_dynamical_dependence_var_detects_hidden_predictive_input():
    """A channel driven by another channel has positive DD when isolated."""
    coefficients = torch.tensor(
        [[[0.40, 0.70], [0.00, 0.20]]], dtype=torch.float64
    )
    covariance = torch.eye(2, dtype=torch.float64)
    system = build_var_system(coefficients, covariance)

    driven = dynamical_dependence(
        system, torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    )
    autonomous = dynamical_dependence(
        system, torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    )
    assert driven.item() > 1e-4
    assert abs(autonomous.item()) < 1e-8


def test_dynamical_dependence_general_state_space_needs_no_state_covariance():
    """DD depends on innovations, not stationary observation covariance/AIS."""
    system = StateSpaceModel(
        transition=torch.tensor([[0.4]], dtype=torch.float64),
        observation=torch.tensor([[1.0], [0.6]], dtype=torch.float64),
        process_covariance=torch.tensor([[0.3]], dtype=torch.float64),
        observation_covariance=torch.tensor(
            [[0.8, 0.1], [0.1, 0.9]], dtype=torch.float64
        ),
        state_covariance=None,
    )
    value = dynamical_dependence(
        system, torch.tensor([[1.0, -0.2]], dtype=torch.float64)
    )
    assert torch.isfinite(value)
    assert value.item() >= -1e-9


def test_dynamical_dependence_default_base_is_bits():
    """The public default remains base two while natural logs remain explicit."""
    system = _iss_fixture()
    projection = torch.tensor([[1.0, 0.35]], dtype=torch.float64)
    bits = dynamical_dependence(system, projection)
    nats = dynamical_dependence(system, projection, base=np.e)
    torch.testing.assert_close(
        bits, nats / np.log(2.0), rtol=1e-10, atol=1e-12
    )


@pytest.mark.parametrize("base", [0.0, 1.0, -2.0, float("inf"), float("nan")])
def test_dynamical_dependence_rejects_invalid_log_base(base):
    """Information units require a finite positive logarithm base other than one."""
    with pytest.raises(ValueError, match="base"):
        dynamical_dependence(
            _iss_fixture(), torch.eye(2, dtype=torch.float64), base=base
        )


def test_dynamical_dependence_rejects_projection_shape_mismatch():
    """Projection columns must match the microscopic observation dimension."""
    with pytest.raises(ValueError, match="observation dimension"):
        dynamical_dependence(
            _iss_fixture(), torch.ones((1, 3), dtype=torch.float64)
        )


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_dynamical_dependence_rejects_nonfinite_projection(bad_value):
    """Projection validation must fail before rank or Riccati operations."""
    projection = torch.tensor([[1.0, bad_value]], dtype=torch.float64)
    with pytest.raises(ValueError, match="finite"):
        dynamical_dependence(_iss_fixture(), projection)


def test_dynamical_dependence_rejects_rank_deficient_projection():
    """The log-determinant definition requires a full-row-rank coarse-graining."""
    projection = torch.tensor(
        [[1.0, 0.0], [2.0, 0.0]], dtype=torch.float64
    )
    with pytest.raises(ValueError, match="full row rank"):
        dynamical_dependence(_iss_fixture(), projection)


def test_stochastic_interaction_matches_independent_innovation_reference():
    """SI must equal the full-vs-marginal innovations log-volume identity."""
    system = StateSpaceModel(
        transition=torch.tensor([[0.55, 0.10], [0.00, 0.40]], dtype=torch.float64),
        observation=torch.tensor([[1.0, 0.2], [0.3, 0.9]], dtype=torch.float64),
        process_covariance=torch.tensor([[0.25, 0.03], [0.03, 0.18]], dtype=torch.float64),
        observation_covariance=torch.tensor([[0.60, 0.08], [0.08, 0.75]], dtype=torch.float64),
    )
    full = _reference_ss_innovations_covariance(system)
    part_0 = _reference_ss_innovations_covariance(system, (0,))
    part_1 = _reference_ss_innovations_covariance(system, (1,))
    expected_nats = (
        np.linalg.slogdet(part_0)[1]
        + np.linalg.slogdet(part_1)[1]
        - np.linalg.slogdet(full)[1]
    )
    actual_nats = stochastic_interaction(
        system, [[0], [1]], base=np.e
    )
    torch.testing.assert_close(
        actual_nats,
        torch.tensor(expected_nats, dtype=torch.float64),
        rtol=1e-8,
        atol=1e-10,
    )
    actual_bits = stochastic_interaction(system, [[0], [1]])
    torch.testing.assert_close(
        actual_bits,
        actual_nats / np.log(2.0),
        rtol=1e-10,
        atol=1e-12,
    )
