import math

import torch

from complextorch import (
    InnovationsStateSpace,
    cmem1_full_past,
    dynamical_dependence,
    modal_decomposition,
    modal_observation_covariance,
    project_innovations_state_space,
)
from complextorch.linalg import spd_logdet
from complextorch.measures.backbone import (
    observation_autocovariances,
    predictive_information_from_model,
)


def _example_system(batch: bool = False) -> InnovationsStateSpace:
    dtype = torch.float64
    transition = torch.tensor(
        [[0.55, 0.18, 0.00], [0.00, 0.34, 0.12], [0.00, 0.00, 0.16]],
        dtype=dtype,
    )
    observation = torch.tensor(
        [[0.55, 0.18, 0.00], [0.00, 0.34, 0.12], [0.00, 0.00, 0.16]],
        dtype=dtype,
    )
    gain = torch.eye(3, dtype=dtype)
    covariance = torch.tensor(
        [[0.40, 0.04, 0.00], [0.04, 0.35, 0.02], [0.00, 0.02, 0.25]],
        dtype=dtype,
    )
    if not batch:
        return InnovationsStateSpace(transition, observation, gain, covariance)
    return InnovationsStateSpace(
        torch.stack((transition, transition * 0.9)),
        torch.stack((observation, observation * 0.9)),
        gain.expand(2, -1, -1).clone(),
        covariance.expand(2, -1, -1).clone(),
    )


def _sorted_modes(decomposition):
    order = torch.argsort(decomposition.poles.real)
    return (
        decomposition.poles[order],
        decomposition.residues[order],
        decomposition.strengths[order],
    )


def test_modal_covariance_matches_canonical_backbone_batched():
    system = _example_system(batch=True)
    modal = modal_decomposition(system)
    reconstructed = modal_observation_covariance(modal)
    expected = observation_autocovariances(system, 0)[:, 0]
    assert reconstructed.shape == expected.shape == (2, 3, 3)
    assert torch.allclose(reconstructed, expected, rtol=1e-11, atol=1e-12)


def test_modal_residues_are_similarity_invariant():
    system = _example_system()
    transform = torch.tensor(
        [[1.0, 0.2, 0.1], [0.0, 1.2, -0.15], [0.1, 0.0, 0.9]],
        dtype=torch.float64,
    )
    identity = torch.eye(3, dtype=torch.float64)
    transform_inverse = torch.linalg.solve(transform, identity)
    similar = InnovationsStateSpace(
        transform @ system.transition @ transform_inverse,
        system.observation @ transform_inverse,
        transform @ system.gain,
        system.innovation_covariance,
    )
    poles, residues, strengths = _sorted_modes(modal_decomposition(system))
    poles_t, residues_t, strengths_t = _sorted_modes(modal_decomposition(similar))
    assert torch.allclose(poles, poles_t, rtol=1e-11, atol=1e-12)
    assert torch.allclose(residues, residues_t, rtol=1e-10, atol=1e-11)
    assert torch.allclose(strengths, strengths_t, rtol=1e-10, atol=1e-11)


def test_fully_observed_var1_residue_identity():
    dtype = torch.float64
    transition = torch.tensor(
        [[0.62, 0.22, 0.00], [0.00, 0.41, 0.13], [0.00, 0.00, 0.19]],
        dtype=dtype,
    )
    system = InnovationsStateSpace(
        transition,
        transition.clone(),
        torch.eye(3, dtype=dtype),
        torch.eye(3, dtype=dtype) / 3.0,
    )
    modal = modal_decomposition(system)
    poles, right = torch.linalg.eig(transition)
    identity = torch.eye(3, dtype=right.dtype)
    left_rows = torch.linalg.solve(right, identity)
    projectors = right.transpose(-1, -2)[:, :, None] * left_rows[:, None, :]
    expected = poles[:, None, None] * projectors
    assert torch.allclose(modal.poles, poles, rtol=1e-12, atol=1e-13)
    assert torch.allclose(modal.residues, expected, rtol=1e-11, atol=1e-12)


def test_nonminimal_distinct_state_mode_is_flagged_inactive():
    base = _example_system()
    dtype = torch.float64
    transition = torch.zeros((4, 4), dtype=dtype)
    transition[:3, :3] = base.transition
    transition[3, 3] = 0.08
    observation = torch.zeros((3, 4), dtype=dtype)
    observation[:, :3] = base.observation
    gain = torch.zeros((4, 3), dtype=dtype)
    gain[:3] = base.gain
    augmented = InnovationsStateSpace(
        transition, observation, gain, base.innovation_covariance
    )
    modal = modal_decomposition(augmented)
    assert int(modal.active.sum()) == 3
    inactive = torch.where(~modal.active)[0]
    assert inactive.numel() == 1
    assert float(modal.strengths[inactive[0]]) < 1e-12
    expected = observation_autocovariances(base, 0)[0]
    assert torch.allclose(
        modal_observation_covariance(modal), expected, rtol=1e-11, atol=1e-12
    )


def test_modal_decomposition_rejects_repeated_poles():
    dtype = torch.float64
    transition = torch.diag(torch.tensor([0.4, 0.4], dtype=dtype))
    system = InnovationsStateSpace(
        transition,
        transition.clone(),
        torch.eye(2, dtype=dtype),
        torch.eye(2, dtype=dtype),
    )
    try:
        modal_decomposition(system)
    except ValueError as exc:
        assert "simple" in str(exc)
    else:
        raise AssertionError("repeated poles must be rejected")


def test_dd_projected_pi_identity():
    system = _example_system()
    projection = torch.tensor([[0.6, -0.2, 0.7]], dtype=torch.float64)
    reduced = project_innovations_state_space(system, projection)
    gamma0 = observation_autocovariances(system, 0)[0]
    gamma_l = projection @ gamma0 @ projection.T
    full_history_v = projection @ system.innovation_covariance @ projection.T
    pi_l = predictive_information_from_model(reduced, base=2.0)
    dd = dynamical_dependence(system, projection, base=2.0)
    rhs = (spd_logdet(gamma_l) - spd_logdet(full_history_v)) / math.log(2.0)
    assert torch.allclose(2.0 * pi_l + dd, rhs, rtol=1e-10, atol=1e-11)


def test_normalized_information_capabilities_ignore_global_innovation_scale():
    system = _example_system()
    scaled = InnovationsStateSpace(
        system.transition,
        system.observation,
        system.gain,
        7.5 * system.innovation_covariance,
    )
    projection = torch.tensor([[0.6, -0.2, 0.7]], dtype=torch.float64)
    values = torch.stack(
        (
            predictive_information_from_model(system, base=2.0),
            cmem1_full_past(system, base=2.0),
            dynamical_dependence(system, projection, base=2.0),
        )
    )
    scaled_values = torch.stack(
        (
            predictive_information_from_model(scaled, base=2.0),
            cmem1_full_past(scaled, base=2.0),
            dynamical_dependence(scaled, projection, base=2.0),
        )
    )
    assert torch.allclose(values, scaled_values, rtol=1e-9, atol=1e-10)


def test_modal_decomposition_preserves_float32_precision_family():
    dtype = torch.float32
    transition = torch.tensor([[0.5, 0.1], [0.0, 0.2]], dtype=dtype)
    system = InnovationsStateSpace(
        transition,
        transition.clone(),
        torch.eye(2, dtype=dtype),
        torch.eye(2, dtype=dtype) / 2.0,
    )
    modal = modal_decomposition(system)
    assert modal.poles.dtype == torch.complex64
    assert modal.residues.dtype == torch.complex64
    assert modal.strengths.dtype == torch.float32
    assert modal.innovation_covariance.dtype == torch.float32
    assert modal_observation_covariance(modal).dtype == torch.float32
