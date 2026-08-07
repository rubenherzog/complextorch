import torch

from complextorch import (
    InnovationsStateSpace,
    optimise_dynamical_dependence_proxy_riemannian,
    optimise_dynamical_dependence_spectral_riemannian,
    orthonormalise_projection,
)


def _system(dtype=torch.float64):
    a = torch.tensor(
        [[0.42, 0.08, 0.00], [0.00, 0.31, 0.06], [0.02, 0.00, 0.24]],
        dtype=dtype,
    )
    c = torch.tensor(
        [[1.0, 0.2, 0.0], [0.1, 0.8, 0.15], [0.0, -0.1, 0.9]],
        dtype=dtype,
    )
    k = torch.tensor(
        [[0.30, 0.04, 0.00], [0.02, 0.24, 0.03], [0.00, 0.05, 0.20]],
        dtype=dtype,
    )
    factor = torch.tensor(
        [[1.20, 0.00, 0.00], [0.25, 0.90, 0.00], [0.10, -0.12, 1.10]],
        dtype=dtype,
    )
    gain_t = torch.linalg.solve_triangular(
        factor.transpose(-1, -2), k.transpose(-1, -2), upper=True
    )
    return InnovationsStateSpace(
        a,
        factor @ c,
        gain_t.transpose(-1, -2),
        factor @ factor.transpose(-1, -2),
    )


def _initializations(dtype=torch.float64):
    raw = torch.tensor(
        [
            [[0.6, -0.3, 0.7]],
            [[0.2, 0.9, -0.3]],
            [[-0.8, 0.1, 0.5]],
            [[0.3, -0.7, -0.4]],
        ],
        dtype=dtype,
    )
    return orthonormalise_projection(raw)


def _projector(matrix):
    return matrix.transpose(-1, -2) @ matrix


def _stack_independent(results):
    order = torch.argsort(torch.cat([result.objective for result in results]), stable=True)
    return {
        "objective": torch.cat([result.objective for result in results])[order],
        "projection": torch.cat([result.projection for result in results])[order],
        "convergence": torch.cat([result.convergence for result in results])[order],
        "step_size": torch.cat([result.step_size for result in results])[order],
        "iterations": torch.cat([result.iterations for result in results])[order],
        "objective_evaluations": torch.cat(
            [result.objective_evaluations for result in results]
        )[order],
        "gradient_evaluations": torch.cat(
            [result.gradient_evaluations for result in results]
        )[order],
        "backtracking_evaluations": torch.cat(
            [result.backtracking_evaluations for result in results]
        )[order],
        "history": torch.cat([result.history for result in results], dim=0)[order],
    }


def _assert_same_search(batched, independent):
    torch.testing.assert_close(
        batched.objective, independent["objective"], rtol=1e-11, atol=1e-13
    )
    torch.testing.assert_close(
        _projector(batched.projection),
        _projector(independent["projection"]),
        rtol=2e-10,
        atol=2e-12,
    )
    assert torch.equal(batched.convergence, independent["convergence"])
    torch.testing.assert_close(
        batched.step_size, independent["step_size"], rtol=0.0, atol=0.0
    )
    assert torch.equal(batched.iterations, independent["iterations"])
    assert torch.equal(
        batched.objective_evaluations, independent["objective_evaluations"]
    )
    assert torch.equal(
        batched.gradient_evaluations, independent["gradient_evaluations"]
    )
    assert torch.equal(
        batched.backtracking_evaluations,
        independent["backtracking_evaluations"],
    )
    torch.testing.assert_close(
        batched.history,
        independent["history"],
        rtol=1e-11,
        atol=1e-13,
        equal_nan=True,
    )


def test_proxy_batched_armijo_matches_independent_restarts_exactly():
    system = _system()
    initial = _initializations()
    kwargs = dict(
        max_iterations=40,
        initial_step_size=2.0,
        gradient_tolerance=1e-12,
        objective_tolerance=1e-14,
        history=True,
    )
    batched = optimise_dynamical_dependence_proxy_riemannian(
        system, initial, **kwargs
    )
    independent = _stack_independent(
        [
            optimise_dynamical_dependence_proxy_riemannian(
                system, initial[index], **kwargs
            )
            for index in range(initial.shape[0])
        ]
    )
    _assert_same_search(batched, independent)


def test_spectral_batched_armijo_matches_independent_restarts_exactly():
    system = _system()
    initial = _initializations()
    frequencies = torch.linspace(0.0, 0.5, 33, dtype=torch.float64)
    kwargs = dict(
        max_iterations=25,
        initial_step_size=1.0,
        gradient_tolerance=1e-12,
        objective_tolerance=1e-14,
        history=True,
    )
    batched = optimise_dynamical_dependence_spectral_riemannian(
        system, initial, frequencies, **kwargs
    )
    independent = _stack_independent(
        [
            optimise_dynamical_dependence_spectral_riemannian(
                system, initial[index], frequencies, **kwargs
            )
            for index in range(initial.shape[0])
        ]
    )
    _assert_same_search(batched, independent)


def test_batched_runs_remain_row_orthonormal():
    system = _system(dtype=torch.float32)
    initial = _initializations(dtype=torch.float32)
    result = optimise_dynamical_dependence_proxy_riemannian(
        system, initial, max_iterations=12, initial_step_size=2.0
    )
    identity = torch.eye(initial.shape[-2], dtype=torch.float32).expand(
        initial.shape[0], -1, -1
    )
    torch.testing.assert_close(
        result.projection @ result.projection.transpose(-1, -2),
        identity,
        rtol=2e-5,
        atol=2e-6,
    )
    assert bool(torch.isfinite(result.objective).all())
    assert bool(torch.isfinite(result.projection).all())
