import torch

from complextorch import (
    DDGradientSearchResult,
    DDOptimizationResult,
    DDRiemannianSearchResult,
    InnovationsStateSpace,
    optimise_dynamical_dependence,
    optimise_dynamical_dependence_proxy,
    optimise_dynamical_dependence_proxy_riemannian,
    optimise_dynamical_dependence_spectral,
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


def _assert_common_matches_legacy(common, legacy):
    torch.testing.assert_close(common.objective, legacy.objective)
    torch.testing.assert_close(common.projection, legacy.projection)
    torch.testing.assert_close(common.step_size, legacy.step_size)
    assert torch.equal(common.iterations, legacy.iterations)
    assert torch.equal(common.convergence, legacy.convergence)


def test_unified_default_is_complexbox_and_matches_proxy_legacy_exactly():
    system = _system()
    initial = _initial()
    kwargs = dict(
        lags=5,
        max_iterations=25,
        history=True,
    )
    legacy = optimise_dynamical_dependence_proxy(
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

    assert isinstance(legacy, DDGradientSearchResult)
    assert isinstance(common, DDOptimizationResult)
    assert common.optimizer == "complexbox"
    assert common.objective_name == "proxy"
    _assert_common_matches_legacy(common, legacy)
    assert torch.equal(common.objective_evaluations, legacy.iterations)
    assert torch.equal(common.gradient_evaluations, legacy.iterations)
    assert bool(torch.all(common.backtracking_evaluations == 0))
    assert common.history is not None
    assert common.history.shape[-1] == 4
    torch.testing.assert_close(common.history[..., :3], legacy.history, equal_nan=True)


def test_unified_complexbox_matches_spectral_legacy_exactly():
    system = _system()
    initial = _initial()
    frequencies = torch.linspace(0.0, 0.5, 33, dtype=torch.float64)
    legacy = optimise_dynamical_dependence_spectral(
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
    assert common.optimizer == "complexbox"
    assert common.objective_name == "spectral"
    _assert_common_matches_legacy(common, legacy)


def test_unified_riemannian_matches_proxy_legacy_exactly():
    system = _system()
    initial = _initial()
    options = {
        "initial_step_size": 1.0,
        "gradient_tolerance": 1e-12,
        "objective_tolerance": 1e-14,
    }
    legacy = optimise_dynamical_dependence_proxy_riemannian(
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

    assert isinstance(legacy, DDRiemannianSearchResult)
    assert common.optimizer == "riemannian_armijo"
    _assert_common_matches_legacy(common, legacy)
    assert torch.equal(common.objective_evaluations, legacy.objective_evaluations)
    assert torch.equal(common.gradient_evaluations, legacy.gradient_evaluations)
    assert torch.equal(
        common.backtracking_evaluations, legacy.backtracking_evaluations
    )
    torch.testing.assert_close(common.history, legacy.history, equal_nan=True)


def test_unified_riemannian_matches_spectral_legacy_exactly():
    system = _system()
    initial = _initial()
    frequencies = torch.linspace(0.0, 0.5, 33, dtype=torch.float64)
    options = {"initial_step_size": 1.0}
    legacy = optimise_dynamical_dependence_spectral_riemannian(
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
    _assert_common_matches_legacy(common, legacy)


def test_legacy_public_return_types_remain_backend_specific():
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

    try:
        optimise_dynamical_dependence(
            system,
            initial,
            objective="proxy",
            frequencies=frequencies,
        )
    except ValueError as exc:
        assert "frequencies" in str(exc)
    else:
        raise AssertionError("proxy objective accepted frequencies")

    try:
        optimise_dynamical_dependence(
            system,
            initial,
            objective="spectral",
            lags=3,
            frequencies=frequencies,
        )
    except ValueError as exc:
        assert "lags" in str(exc)
    else:
        raise AssertionError("spectral objective accepted lags")


def test_unified_api_protects_common_arguments_from_backend_options():
    system = _system()
    initial = _initial()
    try:
        optimise_dynamical_dependence(
            system,
            initial,
            objective="proxy",
            max_iterations=5,
            optimizer_options={"max_iterations": 7},
        )
    except ValueError as exc:
        assert "max_iterations" in str(exc)
    else:
        raise AssertionError("optimizer_options overrode common max_iterations")
