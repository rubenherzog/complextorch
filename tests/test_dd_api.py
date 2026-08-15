import inspect

import pytest
import torch

import complextorch
from complextorch import (
    DDGradientSearchResult,
    DDOptimizationResult,
    InnovationsStateSpace,
    optimise_dynamical_dependence,
    optimise_dynamical_dependence_proxy,
    optimise_dynamical_dependence_spectral,
    orthonormalise_projection,
)
from complextorch.dd_optimization import (
    DDRiemannianSearchResult,
    optimise_dynamical_dependence_proxy_riemannian,
    optimise_dynamical_dependence_spectral_riemannian,
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


def _initial(dtype=torch.float64):
    raw = torch.tensor(
        [
            [[0.6, -0.3, 0.7]],
            [[0.2, 0.9, -0.3]],
            [[-0.8, 0.1, 0.5]],
        ],
        dtype=dtype,
    )
    return orthonormalise_projection(raw)


def _assert_common_matches_backend(common, backend):
    torch.testing.assert_close(common.objective, backend.objective)
    torch.testing.assert_close(common.projection, backend.projection)
    torch.testing.assert_close(common.step_size, backend.step_size)
    assert torch.equal(common.iterations, backend.iterations)
    assert torch.equal(common.convergence, backend.convergence)


def test_root_api_is_unified_and_complexbox_legacy_is_preserved():
    assert hasattr(complextorch, "optimise_dynamical_dependence")
    assert hasattr(complextorch, "optimise_dynamical_dependence_proxy")
    assert hasattr(complextorch, "optimise_dynamical_dependence_spectral")
    assert not hasattr(
        complextorch, "optimise_dynamical_dependence_proxy_riemannian"
    )
    assert not hasattr(
        complextorch, "optimise_dynamical_dependence_spectral_riemannian"
    )
    assert not hasattr(complextorch, "DDRiemannianSearchResult")


def test_unified_dd_optimizer_defaults_to_complexbox():
    signature = inspect.signature(optimise_dynamical_dependence)
    assert signature.parameters["optimizer"].default == "adaptive"


def test_unified_default_matches_proxy_complexbox_exactly():
    system = _system()
    initial = _initial()
    kwargs = dict(lags=5, max_iterations=25, history=True)
    backend = optimise_dynamical_dependence_proxy(
        system,
        initial,
        variant=1,
        initial_step_size=1e-3,
        **kwargs,
    )
    common = optimise_dynamical_dependence(
        system,
        initial,
        objective="proxy",
        optimizer_options={"variant": 1, "initial_step_size": 1e-3},
        **kwargs,
    )

    assert isinstance(backend, DDGradientSearchResult)
    assert isinstance(common, DDOptimizationResult)
    assert common.optimizer == "adaptive"
    assert common.objective_name == "proxy"
    _assert_common_matches_backend(common, backend)
    assert torch.equal(common.objective_evaluations, backend.iterations)
    assert torch.equal(common.gradient_evaluations, backend.iterations)
    assert bool(torch.all(common.backtracking_evaluations == 0))
    assert common.history is not None
    assert common.history.shape[-1] == 4
    torch.testing.assert_close(
        common.history[..., :3], backend.history, equal_nan=True
    )


def test_unified_complexbox_matches_spectral_backend_exactly():
    system = _system()
    initial = _initial()
    frequencies = torch.linspace(0.0, 0.5, 33, dtype=torch.float64)
    backend = optimise_dynamical_dependence_spectral(
        system,
        initial,
        frequencies,
        max_iterations=20,
        variant=1,
        initial_step_size=1e-3,
    )
    common = optimise_dynamical_dependence(
        system,
        initial,
        objective="spectral",
        frequencies=frequencies,
        max_iterations=20,
        optimizer_options={"variant": 1, "initial_step_size": 1e-3},
    )
    assert common.optimizer == "adaptive"
    assert common.objective_name == "spectral"
    _assert_common_matches_backend(common, backend)


def test_unified_riemannian_matches_proxy_backend_exactly():
    system = _system()
    initial = _initial()
    options = {
        "initial_step_size": 1.0,
        "gradient_tolerance": 1e-12,
        "objective_tolerance": 1e-14,
    }
    backend = optimise_dynamical_dependence_proxy_riemannian(
        system,
        initial,
        max_iterations=25,
        history=True,
        **options,
    )
    common = optimise_dynamical_dependence(
        system,
        initial,
        objective="proxy",
        optimizer="riemannian_armijo",
        max_iterations=25,
        history=True,
        optimizer_options=options,
    )

    assert isinstance(backend, DDRiemannianSearchResult)
    assert common.optimizer == "armijo"
    _assert_common_matches_backend(common, backend)
    assert torch.equal(common.objective_evaluations, backend.objective_evaluations)
    assert torch.equal(common.gradient_evaluations, backend.gradient_evaluations)
    assert torch.equal(
        common.backtracking_evaluations, backend.backtracking_evaluations
    )
    torch.testing.assert_close(common.history, backend.history, equal_nan=True)


def test_unified_riemannian_matches_spectral_backend_exactly():
    system = _system()
    initial = _initial()
    frequencies = torch.linspace(0.0, 0.5, 33, dtype=torch.float64)
    options = {"initial_step_size": 1.0}
    backend = optimise_dynamical_dependence_spectral_riemannian(
        system,
        initial,
        frequencies,
        max_iterations=15,
        **options,
    )
    common = optimise_dynamical_dependence(
        system,
        initial,
        objective="spectral",
        optimizer="riemannian_armijo",
        frequencies=frequencies,
        max_iterations=15,
        optimizer_options=options,
    )
    _assert_common_matches_backend(common, backend)


def test_backend_specific_return_types_remain_separate():
    system = _system()
    initial = _initial()[0]
    complexbox = optimise_dynamical_dependence_proxy(
        system, initial, max_iterations=4
    )
    riemannian = optimise_dynamical_dependence_proxy_riemannian(
        system, initial, max_iterations=4
    )
    assert isinstance(complexbox, DDGradientSearchResult)
    assert isinstance(riemannian, DDRiemannianSearchResult)


def test_unified_api_rejects_cross_objective_parameters():
    system = _system()
    initial = _initial()
    frequencies = torch.linspace(0.0, 0.5, 9, dtype=torch.float64)

    with pytest.raises(ValueError, match="frequencies"):
        optimise_dynamical_dependence(
            system,
            initial,
            objective="proxy",
            frequencies=frequencies,
        )
    with pytest.raises(ValueError, match="lags"):
        optimise_dynamical_dependence(
            system,
            initial,
            objective="spectral",
            lags=3,
            frequencies=frequencies,
        )
    with pytest.raises(ValueError, match="frequencies is required"):
        optimise_dynamical_dependence(
            system,
            initial,
            objective="spectral",
        )


def test_unified_api_protects_common_arguments_from_backend_options():
    system = _system()
    initial = _initial()
    with pytest.raises(ValueError, match="max_iterations"):
        optimise_dynamical_dependence(
            system,
            initial,
            objective="proxy",
            max_iterations=5,
            optimizer_options={"max_iterations": 7},
        )


def test_unified_api_rejects_unknown_objective_optimizer_and_backend_option():
    system = _system()
    initial = _initial()
    with pytest.raises(ValueError, match="objective"):
        optimise_dynamical_dependence(
            system, initial, objective="unknown"  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="optimizer"):
        optimise_dynamical_dependence(
            system,
            initial,
            objective="proxy",
            optimizer="unknown",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError):
        optimise_dynamical_dependence(
            system,
            initial,
            objective="proxy",
            optimizer_options={"not_an_option": 1},
        )
