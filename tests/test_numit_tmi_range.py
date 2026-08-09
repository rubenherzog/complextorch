import torch

from complextorch import build_var_system
from complextorch.numit import (
    _match_tmi_by_spectral_radius,
    _numerical_radius_bounds,
    _random_var_shapes,
    _var_decay_to_radius,
    _wishart_identity,
    var_total_mutual_information,
)


def _random_null_inputs(dtype: torch.dtype, *, batch: int = 8):
    generator = torch.Generator().manual_seed(2026)
    coefficients = _random_var_shapes(
        batch,
        4,
        2,
        generator=generator,
        dtype=dtype,
        device=torch.device("cpu"),
    )
    covariance = _wishart_identity(
        batch,
        4,
        generator=generator,
        dtype=dtype,
        device=torch.device("cpu"),
    )
    return coefficients, covariance


def test_float64_radius_bounds_expand_far_beyond_fixed_1e5_margin():
    low, high = _numerical_radius_bounds(torch.float64, device=torch.device("cpu"))
    assert float(low) < 1e-10
    assert float(high) > 1.0 - 1e-10
    assert 0.0 < float(low) < float(high) < 1.0


def test_float32_radius_bounds_remain_dtype_safe_near_unit_root():
    low, high = _numerical_radius_bounds(torch.float32, device=torch.device("cpu"))
    assert 0.0 < float(low) < 1e-5
    assert 1.0 - 1e-5 < float(high) < 1.0


def test_float64_matches_tmi_across_eight_orders_of_magnitude_and_high_values():
    coefficients, covariance = _random_null_inputs(torch.float64)
    target = torch.tensor(
        [1e-8, 1e-5, 1e-3, 5e-2, 0.5, 2.0, 8.0, 20.0],
        dtype=torch.float64,
    )
    model, radius = _match_tmi_by_spectral_radius(
        coefficients,
        covariance,
        target,
        base=2.0,
    )
    achieved = var_total_mutual_information(model)
    torch.testing.assert_close(achieved, target, rtol=2e-7, atol=1e-10)
    assert bool(((radius > 0) & (radius < 1)).all())
    assert float(radius.max()) > 0.9999


def test_float32_matches_broad_practical_tmi_range_with_expected_precision():
    coefficients, covariance = _random_null_inputs(torch.float32)
    target = torch.tensor(
        [1e-5, 1e-3, 1e-2, 5e-2, 0.5, 2.0, 5.0, 8.0],
        dtype=torch.float32,
    )
    model, radius = _match_tmi_by_spectral_radius(
        coefficients,
        covariance,
        target,
        base=2.0,
    )
    achieved = var_total_mutual_information(model)
    torch.testing.assert_close(achieved, target, rtol=5e-5, atol=3e-5)
    assert bool(((radius > 0) & (radius < 1)).all())


def test_high_float64_radius_supports_tmi_well_above_old_ceiling():
    coefficients, covariance = _random_null_inputs(torch.float64, batch=16)
    target = torch.full((16,), 15.0, dtype=torch.float64)
    model, radius = _match_tmi_by_spectral_radius(
        coefficients,
        covariance,
        target,
        base=2.0,
    )
    achieved = var_total_mutual_information(model)
    torch.testing.assert_close(achieved, target, rtol=1e-8, atol=1e-9)
    assert bool((radius > 0.999).any())


def test_tmi_remains_monotone_across_extreme_float64_radius_grid():
    coefficients, covariance = _random_null_inputs(torch.float64, batch=16)
    radii = torch.tensor(
        [
            1e-12,
            1e-9,
            1e-6,
            1e-4,
            1e-2,
            0.1,
            0.5,
            0.9,
            0.99,
            0.999,
            0.9999,
            0.99999,
            0.999999,
            0.99999999,
            1.0 - 1e-10,
            1.0 - 1e-12,
            1.0 - 2e-14,
        ],
        dtype=torch.float64,
    )
    curve = []
    for radius in radii:
        model = build_var_system(
            _var_decay_to_radius(
                coefficients,
                radius.expand(coefficients.shape[0]),
            ),
            covariance,
        )
        curve.append(var_total_mutual_information(model))
    curve = torch.stack(curve)
    increments = curve[1:] - curve[:-1]
    assert bool(torch.isfinite(curve).all())
    assert bool((increments >= -1e-10).all())
    assert float(curve[-1].min()) > 20.0
