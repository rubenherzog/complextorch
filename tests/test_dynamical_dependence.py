"""Equation-level regression tests for dynamical dependence."""

import numpy as np
import torch
from scipy.linalg import solve_discrete_are

from complextorch import (
    InnovationsStateSpace,
    StateSpaceModel,
    build_var_system,
    dynamical_dependence,
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
    return float(np.linalg.slogdet(reduced_v)[1] - np.linalg.slogdet(reduced_r)[1])


def test_dynamical_dependence_matches_independent_dare_reference():
    """DD must equal the exact reduced-DARE log-determinant ratio."""
    system = _iss_fixture()
    projection = torch.tensor([[1.0, 0.35]], dtype=torch.float64)
    expected = _reference_dd(system, projection)
    actual = dynamical_dependence(system, projection)
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=actual.dtype), rtol=1e-8, atol=1e-10)


def test_dynamical_dependence_identity_projection_is_zero():
    """An invertible full-dimensional macro variable retains the full past."""
    system = _iss_fixture()
    projection = torch.eye(2, dtype=torch.float64)
    actual = dynamical_dependence(system, projection)
    torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0.0, atol=1e-10)


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
    """DD depends on innovations, not the stationary observation covariance/AIS."""
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


def test_dynamical_dependence_log_base_only_changes_units():
    """Natural-log and base-2 DD differ only by the expected unit conversion."""
    system = _iss_fixture()
    projection = torch.tensor([[1.0, 0.35]], dtype=torch.float64)
    nats = dynamical_dependence(system, projection)
    bits = dynamical_dependence(system, projection, base=2.0)
    torch.testing.assert_close(bits, nats / np.log(2.0), rtol=1e-10, atol=1e-12)
