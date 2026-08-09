import pytest
import torch

from complextorch import build_var_system
from complextorch.measures.pid import gaussian_pid, gaussian_pid_from_var
from complextorch.numit import (
    _empirical_mid_quantile,
    _var_decay_to_radius,
    numit_pid_var,
    var_total_mutual_information,
)
from complextorch.representations import companion_matrix


def test_gaussian_pid_mmi_matches_defining_equations_and_batches():
    covariance = torch.tensor(
        [[1.0, 0.2, 0.5], [0.2, 1.0, 0.35], [0.5, 0.35, 1.0]],
        dtype=torch.float64,
    )
    batched = covariance.expand(3, -1, -1).clone()
    result = gaussian_pid(batched, 1, 1, redundancy="mmi")
    assert result["total"].shape == (3,)
    torch.testing.assert_close(result["reconstruction"], result["total"])
    assert bool((result["redundant"] >= 0).all())
    assert bool((result["unique_source0"] >= -1e-12).all())
    assert bool((result["unique_source1"] >= -1e-12).all())


def test_var_pid_total_equals_gaussian_var_past_future_tmi_for_var2():
    coefficients = torch.tensor(
        [
            [[0.28, 0.05], [0.04, 0.24]],
            [[0.08, 0.00], [0.01, 0.06]],
        ],
        dtype=torch.float64,
    )
    covariance = torch.tensor([[1.0, 0.1], [0.1, 0.8]], dtype=torch.float64)
    model = build_var_system(coefficients, covariance)
    pid = gaussian_pid_from_var(model, (0,), (1,), redundancy="mmi")
    tmi = var_total_mutual_information(model)
    torch.testing.assert_close(pid["total"], tmi, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(pid["reconstruction"], pid["total"])


def test_var_decay_matches_requested_companion_radius_for_var2_batch():
    generator = torch.Generator().manual_seed(3)
    coefficients = torch.randn((4, 2, 3, 3), generator=generator, dtype=torch.float64)
    target = torch.tensor([0.2, 0.4, 0.65, 0.8], dtype=torch.float64)
    scaled = _var_decay_to_radius(coefficients, target)
    radius = torch.linalg.eigvals(companion_matrix(scaled)).abs().amax(-1)
    torch.testing.assert_close(radius, target, rtol=1e-10, atol=1e-10)


def test_numit_var_matches_tmi_and_returns_reproducible_quantiles():
    coefficients = torch.tensor(
        [[[0.38, 0.12], [0.07, 0.31]]], dtype=torch.float64
    )
    covariance = torch.tensor([[1.0, 0.15], [0.15, 0.9]], dtype=torch.float64)
    model = build_var_system(coefficients, covariance)
    first = numit_pid_var(model, (0,), (1,), n_null=24, seed=7)
    second = numit_pid_var(model, (0,), (1,), n_null=24, seed=7)
    torch.testing.assert_close(
        first.null_tmi,
        first.target_tmi.expand_as(first.null_tmi),
        rtol=1e-7,
        atol=1e-8,
    )
    for name in ("redundant", "unique_source0", "unique_source1", "synergistic"):
        assert 0.0 <= float(first.quantiles[name]) <= 1.0
        torch.testing.assert_close(first.quantiles[name], second.quantiles[name])
        torch.testing.assert_close(first.null_atoms[name], second.null_atoms[name])


def test_numit_observed_pid_is_invariant_to_global_innovation_scale():
    coefficients = torch.tensor(
        [[[0.32, 0.08], [0.05, 0.27]]], dtype=torch.float64
    )
    covariance = torch.tensor([[1.0, 0.2], [0.2, 0.7]], dtype=torch.float64)
    one = build_var_system(coefficients, covariance)
    two = build_var_system(coefficients, 13.0 * covariance)
    pid_one = gaussian_pid_from_var(one, (0,), (1,))
    pid_two = gaussian_pid_from_var(two, (0,), (1,))
    for name in ("redundant", "unique_source0", "unique_source1", "synergistic", "total"):
        torch.testing.assert_close(pid_one[name], pid_two[name], rtol=1e-9, atol=1e-10)


def test_numit_mid_quantile_matches_reference_tie_rule():
    samples = torch.tensor([0.0, 1.0, 1.0, 2.0], dtype=torch.float64)
    value = torch.tensor(1.0, dtype=torch.float64)
    torch.testing.assert_close(
        _empirical_mid_quantile(samples, value),
        torch.tensor(0.5, dtype=torch.float64),
    )


def test_numit_mmi_uses_pooled_unique_null_distribution():
    coefficients = torch.tensor(
        [[[0.38, 0.12], [0.07, 0.31]]], dtype=torch.float64
    )
    covariance = torch.tensor([[1.0, 0.15], [0.15, 0.9]], dtype=torch.float64)
    model = build_var_system(coefficients, covariance)
    result = numit_pid_var(model, (0,), (1,), n_null=24, seed=9, redundancy="mmi")
    pooled = result.null_atoms["unique_source0"] + result.null_atoms["unique_source1"]
    torch.testing.assert_close(result.quantile_samples["unique_source0"], pooled)
    torch.testing.assert_close(result.quantile_samples["unique_source1"], pooled)
    for name in ("unique_source0", "unique_source1"):
        expected = _empirical_mid_quantile(pooled, result.observed[name])
        torch.testing.assert_close(result.quantiles[name], expected)


def test_numit_var_rejects_unequal_source_partition_for_reference_parity():
    coefficients = torch.diag(torch.tensor([0.2, 0.3, 0.4], dtype=torch.float64)).unsqueeze(0)
    model = build_var_system(coefficients, torch.eye(3, dtype=torch.float64))
    with pytest.raises(ValueError, match="equal-sized"):
        numit_pid_var(model, (0,), (1, 2), n_null=4, seed=1)


def test_numit_var2_matches_target_tmi():
    coefficients = torch.tensor(
        [
            [[0.24, 0.05], [0.03, 0.21]],
            [[0.07, 0.01], [0.00, 0.05]],
        ],
        dtype=torch.float64,
    )
    covariance = torch.tensor([[1.0, 0.12], [0.12, 0.85]], dtype=torch.float64)
    model = build_var_system(coefficients, covariance)
    result = numit_pid_var(model, (0,), (1,), n_null=16, seed=12)
    torch.testing.assert_close(
        result.null_tmi,
        result.target_tmi.expand_as(result.null_tmi),
        rtol=1e-7,
        atol=1e-8,
    )


def test_gaussian_pid_dep_and_ccs_reconstruct_total():
    covariance = torch.tensor(
        [[1.0, 0.15, 0.45], [0.15, 1.0, 0.30], [0.45, 0.30, 1.0]],
        dtype=torch.float64,
    )
    for redundancy in ("dep", "ccs"):
        result = gaussian_pid(
            covariance,
            1,
            1,
            redundancy=redundancy,
            ccs_qmc_samples=256,
        )
        torch.testing.assert_close(result["reconstruction"], result["total"])
        assert bool(torch.isfinite(result["redundant"]))
