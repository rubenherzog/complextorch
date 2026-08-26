import torch

from complextorch import (
    capability_mobility,
    finite_difference_jacobian,
    jacobian_rank,
    neutral_projector,
    optimise_prescribed_capabilities,
    pareto_nondominated,
    project_to_capability_level_set,
)


def _capabilities(parameters: torch.Tensor) -> torch.Tensor:
    flat = parameters.reshape(parameters.shape[0], -1)
    x, y, z = flat[:, 0], flat[:, 1], flat[:, 2]
    return torch.stack((x * x + y, x - y * y + z), dim=-1)


def test_finite_difference_jacobian_batches_all_designs():
    dtype = torch.float64
    designs = torch.tensor([[0.4, -0.2, 0.1], [0.7, 0.3, -0.4]], dtype=dtype)
    jacobian = finite_difference_jacobian(
        _capabilities, designs, step=1e-6, batched=True
    )
    expected = torch.stack(
        (
            torch.tensor([[0.8, 1.0, 0.0], [1.0, 0.4, 1.0]], dtype=dtype),
            torch.tensor([[1.4, 1.0, 0.0], [1.0, -0.6, 1.0]], dtype=dtype),
        )
    )
    assert jacobian.shape == (2, 2, 3)
    assert torch.allclose(jacobian, expected, rtol=1e-8, atol=1e-9)


def test_neutral_projector_and_capability_mobility_are_batch_safe():
    target = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    free = torch.tensor([[0.0, 2.0, 3.0]], dtype=torch.float64)
    projector = neutral_projector(target)
    mobility = capability_mobility(free, target)
    assert int(jacobian_rank(target)) == 1
    assert torch.allclose(target @ projector, torch.zeros_like(target), atol=1e-12)
    assert torch.allclose(projector @ projector, projector, atol=1e-12)
    assert torch.allclose(mobility, free, atol=1e-12)


def test_level_set_projection_handles_multiple_starts_in_one_batch():
    dtype = torch.float64

    def capability(parameters):
        return parameters.reshape(parameters.shape[0], -1).sum(-1, keepdim=True)

    initial = torch.tensor(
        [[2.0, 0.0], [0.0, 2.0], [-1.0, 3.0]], dtype=dtype
    )
    result = project_to_capability_level_set(
        initial,
        capability,
        torch.tensor([1.0], dtype=dtype),
        batched=True,
        tolerance=1e-9,
    )
    assert bool(result.converged.all())
    assert float(result.max_error.max()) < 1e-9
    assert torch.allclose(
        result.capabilities, torch.ones((3, 1), dtype=dtype), atol=1e-9
    )


def test_prescribed_capability_optimizer_preserves_multistart_shape():
    dtype = torch.float64

    def capability(parameters):
        flat = parameters.reshape(parameters.shape[0], -1)
        return flat.sum(-1, keepdim=True)

    def objective(parameters, _capabilities):
        return parameters.square().reshape(parameters.shape[0], -1).sum(-1)

    initial = torch.tensor(
        [[1.5, -0.2], [-0.3, 1.7], [2.0, 0.5]], dtype=dtype
    )
    result = optimise_prescribed_capabilities(
        initial,
        capability,
        torch.tensor([1.0], dtype=dtype),
        objective_function=objective,
        steps=80,
        learning_rate=0.03,
        constraint_weight=100.0,
        tolerance=1e-8,
    )
    assert result.parameters.shape == initial.shape
    assert result.capabilities.shape == (3, 1)
    assert bool(result.converged.all())
    assert float(result.max_constraint_error.max()) < 1e-8


def test_prescribed_capability_optimizer_supports_per_run_targets():
    dtype = torch.float64

    def capability(parameters):
        return parameters.reshape(parameters.shape[0], -1).sum(-1, keepdim=True)

    initial = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=dtype)
    targets = torch.tensor([[0.5], [1.5]], dtype=dtype)
    result = optimise_prescribed_capabilities(
        initial,
        capability,
        targets,
        steps=30,
        learning_rate=0.02,
        constraint_weight=50.0,
        tolerance=1e-8,
    )
    assert bool(result.converged.all())
    assert torch.allclose(result.capabilities, targets, atol=1e-8)


def test_pareto_orientation_supports_mixed_minimize_and_maximize():
    values = torch.tensor(
        [
            [1.0, 3.0],
            [2.0, 4.0],
            [0.8, 2.0],
            [2.5, 1.0],
            [3.0, 0.5],
        ],
        dtype=torch.float64,
    )
    # Minimize first objective, maximize second.
    mask = pareto_nondominated(values, maximize=[False, True])
    expected = torch.tensor([True, True, True, False, False])
    assert torch.equal(mask, expected)


def test_prescribed_optimizer_composes_with_exact_model_capability():
    from complextorch import InnovationsStateSpace
    from complextorch.measures.backbone import predictive_information_from_model

    dtype = torch.float64
    innovation = torch.eye(2, dtype=dtype) / 2.0

    def capability(transition):
        batch = transition.shape[0]
        model = InnovationsStateSpace(
            transition,
            transition.clone(),
            torch.eye(2, dtype=dtype).expand(batch, -1, -1).clone(),
            innovation.expand(batch, -1, -1).clone(),
        )
        return predictive_information_from_model(model, base=2.0)[:, None]

    def objective(transition, _capabilities):
        return transition.square().sum(dim=(-1, -2))

    initial = torch.tensor(
        [
            [[0.45, 0.08], [0.00, 0.25]],
            [[0.35, 0.12], [0.03, 0.20]],
        ],
        dtype=dtype,
    )
    result = optimise_prescribed_capabilities(
        initial,
        capability,
        torch.tensor([0.14], dtype=dtype),
        objective_function=objective,
        steps=60,
        learning_rate=0.02,
        constraint_weight=1000.0,
        tolerance=1e-8,
    )
    assert bool(result.converged.all())
    assert torch.allclose(
        result.capabilities,
        torch.full((2, 1), 0.14, dtype=dtype),
        atol=1e-8,
    )


def test_design_differential_primitives_preserve_float32_dtype():
    parameters = torch.tensor([0.4, -0.2, 0.1], dtype=torch.float32)
    jacobian = finite_difference_jacobian(
        _capabilities, parameters, step=1e-3
    )
    projector = neutral_projector(jacobian[:1])
    assert jacobian.dtype == torch.float32
    assert projector.dtype == torch.float32
