import pytest
import torch

from complextorch import (
    InnovationsStateSpace,
    as_innovations_state_space,
    build_var_system,
    compute_all_model_measures,
    demo_var,
    model_autocovariances,
    scale_dynamics,
)


def _var():
    coefficients, covariance = demo_var(n_variables=3, order=2)
    return build_var_system(coefficients, covariance)


def test_public_converter_accepts_all_canonical_representations():
    var = _var()
    state_space = var.to_state_space()
    innovations = as_innovations_state_space(var)

    assert as_innovations_state_space(innovations) is innovations
    for system in (var, state_space):
        converted = as_innovations_state_space(system)
        torch.testing.assert_close(
            model_autocovariances(converted, 3),
            model_autocovariances(var, 3),
            rtol=1e-7,
            atol=1e-9,
        )


def test_lambda_one_recovers_empirical_innovations_process():
    innovations = as_innovations_state_space(_var())
    scaled = scale_dynamics(innovations, 1.0)

    torch.testing.assert_close(scaled.transition, innovations.transition)
    torch.testing.assert_close(scaled.observation, innovations.observation)
    torch.testing.assert_close(scaled.gain, innovations.gain, rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(
        scaled.innovation_covariance,
        innovations.innovation_covariance,
        rtol=1e-8,
        atol=1e-10,
    )
    torch.testing.assert_close(
        model_autocovariances(scaled, 4),
        model_autocovariances(innovations, 4),
        rtol=1e-8,
        atol=1e-10,
    )


def test_lambda_zero_removes_temporal_transition_and_remains_measurable():
    scaled = scale_dynamics(_var(), 0.0)
    torch.testing.assert_close(scaled.transition, torch.zeros_like(scaled.transition))
    measures = compute_all_model_measures(scaled)
    torch.testing.assert_close(
        measures["criticality"]["spectral_radius"],
        torch.zeros_like(measures["criticality"]["spectral_radius"]),
    )
    assert bool(torch.isfinite(measures["dynamics"]["predictive_information"]).all())


def test_scale_grid_has_exact_radius_and_system_major_batch_order():
    var = _var()
    innovations = as_innovations_state_space(var)
    radius = torch.linalg.eigvals(innovations.transition).abs().amax(-1)
    lambdas = torch.tensor([0.0, 0.25, 0.75, 1.0], dtype=radius.dtype)

    scaled = scale_dynamics(innovations, lambdas)
    actual = torch.linalg.eigvals(scaled.transition).abs().amax(-1)
    torch.testing.assert_close(actual, lambdas * radius[0], rtol=1e-8, atol=1e-10)
    assert scaled.transition.shape[0] == lambdas.numel()

    batched = InnovationsStateSpace(
        innovations.transition.expand(2, -1, -1).clone(),
        innovations.observation.expand(2, -1, -1).clone(),
        innovations.gain.expand(2, -1, -1).clone(),
        innovations.innovation_covariance.expand(2, -1, -1).clone(),
    )
    scaled_batch = scale_dynamics(batched, lambdas)
    expected = (lambdas * radius[0]).repeat(2)
    actual_batch = torch.linalg.eigvals(scaled_batch.transition).abs().amax(-1)
    torch.testing.assert_close(actual_batch, expected, rtol=1e-8, atol=1e-10)


def test_batch_grid_matches_individual_scaling():
    innovations = as_innovations_state_space(_var())
    lambdas = torch.tensor([0.0, 0.4, 0.8, 1.0], dtype=innovations.transition.dtype)
    batched = scale_dynamics(innovations, lambdas)

    for index, value in enumerate(lambdas):
        individual = scale_dynamics(innovations, value)
        for actual, expected in (
            (batched.transition[index], individual.transition[0]),
            (batched.observation[index], individual.observation[0]),
            (batched.gain[index], individual.gain[0]),
            (batched.innovation_covariance[index], individual.innovation_covariance[0]),
        ):
            torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-10)


def test_var_and_equivalent_state_space_scale_to_same_observable_process():
    var = _var()
    state_space = var.to_state_space()
    lambdas = torch.tensor([0.0, 0.5, 1.0], dtype=var.coefficients.dtype)

    scaled_var = scale_dynamics(var, lambdas)
    scaled_state = scale_dynamics(state_space, lambdas)
    torch.testing.assert_close(
        model_autocovariances(scaled_var, 4),
        model_autocovariances(scaled_state, 4),
        rtol=1e-7,
        atol=1e-9,
    )


def test_scaling_is_invariant_to_latent_similarity_transform():
    innovations = as_innovations_state_space(_var())
    a = innovations.transition[0]
    c = innovations.observation[0]
    k = innovations.gain[0]
    v = innovations.innovation_covariance[0]
    dimension = a.shape[0]
    transform = torch.eye(dimension, dtype=a.dtype)
    transform[0, 1] = 0.2
    transform[1, 0] = -0.1
    inverse = torch.linalg.inv(transform)
    transformed = InnovationsStateSpace(
        transform @ a @ inverse,
        c @ inverse,
        transform @ k,
        v,
    )

    lambdas = torch.tensor([0.2, 0.7, 1.0], dtype=a.dtype)
    original_scaled = scale_dynamics(InnovationsStateSpace(a, c, k, v), lambdas)
    transformed_scaled = scale_dynamics(transformed, lambdas)
    torch.testing.assert_close(
        model_autocovariances(original_scaled, 5),
        model_autocovariances(transformed_scaled, 5),
        rtol=1e-7,
        atol=1e-9,
    )


def test_near_critical_scaling_and_stability_boundary_validation():
    innovations = as_innovations_state_space(_var())
    radius = torch.linalg.eigvals(innovations.transition).abs().amax()
    near_critical = 0.999 / radius
    scaled = scale_dynamics(innovations, near_critical)
    actual = torch.linalg.eigvals(scaled.transition).abs().amax()
    torch.testing.assert_close(actual, torch.tensor(0.999, dtype=actual.dtype))
    assert bool(torch.isfinite(model_autocovariances(scaled, 2)).all())

    with pytest.raises(ValueError, match="strictly stable"):
        scale_dynamics(innovations, 1.0 / radius)


def test_float32_dtype_device_and_invalid_lambda_contract():
    innovations = as_innovations_state_space(_var())
    system = InnovationsStateSpace(
        innovations.transition.float(),
        innovations.observation.float(),
        innovations.gain.float(),
        innovations.innovation_covariance.float(),
    )
    scaled = scale_dynamics(system, torch.tensor([0.0, 0.5], dtype=torch.float32))
    for tensor in (
        scaled.transition,
        scaled.observation,
        scaled.gain,
        scaled.innovation_covariance,
    ):
        assert tensor.dtype == torch.float32
        assert tensor.device == system.transition.device

    with pytest.raises(ValueError, match="non-negative"):
        scale_dynamics(system, -0.1)
    with pytest.raises(ValueError, match="one-dimensional"):
        scale_dynamics(system, torch.ones(2, 2))
