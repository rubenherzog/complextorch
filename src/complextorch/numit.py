"""NuMIT null-model normalisation for Gaussian PID of VAR systems.

This module implements the VAR construction in Liardi et al. (2025) without
changing the existing inference/bootstrap architecture. Random VAR models are
matched to the observed total past--future mutual information by tuning their
companion spectral radius, then PID atoms are expressed as empirical quantiles
within that matched null ensemble.

Reference implementation
------------------------
``alberto-liardi/NuMIT`` commit ``44efc720c963afb011d376aa9682006657f8c3c0``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from .measures.backbone import predictive_information_from_model
from .measures.pid import PIDRedundancy, gaussian_pid_from_var
from .representations import VARSystem, build_var_system, companion_matrix


@dataclass(frozen=True)
class NuMITPIDResult:
    """Observed PID, TMI-matched null distribution, and NuMIT quantiles."""

    observed: Mapping[str, torch.Tensor]
    quantiles: Mapping[str, torch.Tensor]
    null_atoms: Mapping[str, torch.Tensor]
    quantile_samples: Mapping[str, torch.Tensor]
    target_tmi: torch.Tensor
    null_tmi: torch.Tensor
    null_spectral_radius: torch.Tensor
    n_null: int
    seed: int | None
    redundancy: str


def var_total_mutual_information(model: VARSystem, *, base: float = 2.0) -> torch.Tensor:
    r"""Return :math:`I(X_t;X_{t-1:t-p})` for a stationary Gaussian VAR(p)."""
    return predictive_information_from_model(model, base=base)


def _var_decay_to_radius(coefficients: torch.Tensor, target_radius: torch.Tensor) -> torch.Tensor:
    """MVGC ``specnorm``/``var_decay`` scaling, batched and Torch-native."""
    coefficient = torch.as_tensor(coefficients)
    if coefficient.ndim != 4:
        raise ValueError("coefficients must have shape (batch,p,n,n)")
    target = torch.as_tensor(target_radius, dtype=coefficient.dtype, device=coefficient.device)
    if target.ndim == 0:
        target = target.expand(coefficient.shape[0])
    if target.shape != (coefficient.shape[0],):
        raise ValueError("target_radius must be scalar or have shape (batch,)")
    if bool(((target <= 0) | (target >= 1)).any()):
        raise ValueError("target_radius must lie strictly inside (0,1)")
    old = torch.linalg.eigvals(companion_matrix(coefficient)).abs().amax(-1)
    if bool((old <= 0).any()):
        raise ValueError("cannot rescale a zero-spectral-radius VAR")
    factor = target / old
    powers = torch.arange(
        1, coefficient.shape[1] + 1, dtype=coefficient.dtype, device=coefficient.device
    )
    return coefficient * factor[:, None, None, None].pow(powers[None, :, None, None])


def _wishart_identity(
    batch: int,
    dimension: int,
    *,
    generator: torch.Generator,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Sample Wishart(I, dimension+1), matching the NuMIT VAR reference."""
    normal = torch.randn(
        (batch, dimension, dimension + 1),
        generator=generator,
        dtype=dtype,
        device=device,
    )
    return normal @ normal.transpose(-1, -2)


def _random_var_shapes(
    batch: int,
    n_variables: int,
    order: int,
    *,
    generator: torch.Generator,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Draw otherwise-random VAR coefficient shapes before spectral matching."""
    return torch.randn(
        (batch, order, n_variables, n_variables),
        generator=generator,
        dtype=dtype,
        device=device,
    )


def _numerical_radius_bounds(
    dtype: torch.dtype,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return dtype-aware open bounds for stable spectral-radius matching."""
    if not dtype.is_floating_point:
        raise TypeError("NuMIT spectral-radius matching requires floating-point dtype")
    margin = 64.0 * torch.finfo(dtype).eps
    return (
        torch.tensor(margin, dtype=dtype, device=device),
        torch.tensor(1.0 - margin, dtype=dtype, device=device),
    )


def _safe_tmi_at_radius(
    coefficients: torch.Tensor,
    covariance: torch.Tensor,
    radius: torch.Tensor | float,
    *,
    base: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate TMI without allowing one ill-conditioned null to kill its peers."""
    batch = coefficients.shape[0]
    values = torch.full(
        (batch,), float("nan"), dtype=coefficients.dtype, device=coefficients.device
    )
    valid = torch.zeros((batch,), dtype=torch.bool, device=coefficients.device)
    radius_tensor = torch.as_tensor(
        radius, dtype=coefficients.dtype, device=coefficients.device
    )
    if radius_tensor.ndim == 0:
        radius_tensor = radius_tensor.expand(batch)
    if radius_tensor.shape != (batch,):
        raise ValueError("radius must be scalar or have shape (batch,)")

    def evaluate(indices: torch.Tensor) -> None:
        """Evaluate one subset, splitting recursively after numerical failure."""
        if indices.numel() == 0:
            return
        try:
            model = build_var_system(
                _var_decay_to_radius(
                    coefficients.index_select(0, indices),
                    radius_tensor.index_select(0, indices),
                ),
                covariance.index_select(0, indices),
            )
            result = var_total_mutual_information(model, base=base)
            finite = torch.isfinite(result)
            if bool(finite.any()):
                good = indices[finite]
                values[good] = result[finite]
                valid[good] = True
        except (ValueError, RuntimeError, torch.linalg.LinAlgError):
            if indices.numel() == 1:
                return
            half = indices.numel() // 2
            evaluate(indices[:half])
            evaluate(indices[half:])

    evaluate(torch.arange(batch, device=coefficients.device))
    return values, valid


def _tmi_brackets(
    coefficients: torch.Tensor,
    covariance: torch.Tensor,
    target_tmi: torch.Tensor,
    *,
    base: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Find a numerically safe spectral-radius bracket for every null VAR."""
    batch = coefficients.shape[0]
    target = torch.as_tensor(
        target_tmi, dtype=coefficients.dtype, device=coefficients.device
    )
    if target.ndim == 0:
        target = target.expand(batch)
    if target.shape != (batch,):
        raise ValueError("target_tmi must be scalar or have shape (batch,)")
    if bool((target < 0).any()):
        raise ValueError("target_tmi must be nonnegative")

    lower_bound, _ = _numerical_radius_bounds(
        coefficients.dtype, device=coefficients.device
    )
    low = torch.full_like(target, lower_bound.item())
    low_tmi, low_ok = _safe_tmi_at_radius(
        coefficients, covariance, low, base=base
    )
    valid = low_ok & (target >= low_tmi)
    high = torch.full_like(target, float("nan"))
    unresolved = valid.clone()

    eps_margin = 64.0 * torch.finfo(coefficients.dtype).eps
    gaps = (
        1e-1, 5e-2, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6,
        5e-7, 1e-7, 5e-8, 1e-8, 5e-9, 1e-9, 5e-10,
        1e-10, 1e-11, 1e-12, 1e-13,
    )
    for gap in (value for value in gaps if value > eps_margin):
        indices = torch.where(unresolved)[0]
        if indices.numel() == 0:
            break
        radius = torch.full(
            (indices.numel(),),
            1.0 - gap,
            dtype=coefficients.dtype,
            device=coefficients.device,
        )
        values, safe = _safe_tmi_at_radius(
            coefficients.index_select(0, indices),
            covariance.index_select(0, indices),
            radius,
            base=base,
        )
        unsafe = ~safe
        if bool(unsafe.any()):
            failed = indices[unsafe]
            unresolved[failed] = False
            valid[failed] = False
        reached = safe & (values >= target.index_select(0, indices))
        if bool(reached.any()):
            reached_indices = indices[reached]
            high[reached_indices] = 1.0 - gap
            unresolved[reached_indices] = False

    valid = valid & torch.isfinite(high)
    return low, high, valid


def _match_tmi_by_spectral_radius(
    coefficients: torch.Tensor,
    covariance: torch.Tensor,
    target_tmi: torch.Tensor,
    *,
    base: float,
    iterations: int = 64,
) -> tuple[VARSystem, torch.Tensor]:
    """Match random VARs to target TMI using robust batched bisection in rho."""
    batch = coefficients.shape[0]
    target = torch.as_tensor(
        target_tmi, dtype=coefficients.dtype, device=coefficients.device
    )
    if target.ndim == 0:
        target = target.expand(batch)
    low, high, valid = _tmi_brackets(
        coefficients, covariance, target, base=base
    )
    if not bool(valid.all()):
        raise RuntimeError(
            "target TMI is not numerically reachable for "
            f"{int((~valid).sum().item())} sampled null VAR(s)"
        )
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        model = build_var_system(_var_decay_to_radius(coefficients, mid), covariance)
        value = var_total_mutual_information(model, base=base)
        below = value < target
        low = torch.where(below, mid, low)
        high = torch.where(below, high, mid)
    radius = 0.5 * (low + high)
    model = build_var_system(_var_decay_to_radius(coefficients, radius), covariance)
    return model, radius


def _empirical_mid_quantile(
    samples: torch.Tensor,
    value: torch.Tensor,
    *,
    atol: float = 1e-8,
) -> torch.Tensor:
    """Return the NuMIT/``CompQuantile`` empirical mid-quantile."""
    sample = torch.as_tensor(samples)
    observed = torch.as_tensor(value, dtype=sample.dtype, device=sample.device)
    finite = torch.isfinite(sample)
    n_valid = finite.sum(dim=0)
    if bool((n_valid == 0).any()):
        raise RuntimeError("NuMIT quantile has no finite null samples")
    less = finite & (sample < observed - atol)
    equal = finite & ((sample - observed).abs() < atol)
    return (less.sum(dim=0) + 0.5 * equal.sum(dim=0)).to(sample.dtype) / n_valid


def numit_pid_var(
    model: VARSystem,
    source0: tuple[int, ...],
    source1: tuple[int, ...],
    *,
    redundancy: PIDRedundancy | str = "mmi",
    n_null: int = 1000,
    seed: int | None = None,
    base: float = 2.0,
    ccs_qmc_samples: int = 4096,
) -> NuMITPIDResult:
    """NuMIT-normalise Gaussian VAR PID atoms against TMI-matched null VARs."""
    if model.batch_size != 1:
        raise ValueError("numit_pid_var currently accepts one observed VARSystem")
    if n_null < 2:
        raise ValueError("n_null must be at least two")
    if set(source0) & set(source1) or set(source0) | set(source1) != set(range(model.n_variables)):
        raise ValueError("source0 and source1 must be a disjoint partition of all variables")
    if len(source0) != len(source1):
        raise ValueError(
            "NuMIT VAR PID currently follows the reference equal-sized source partition"
        )

    observed = gaussian_pid_from_var(
        model,
        source0,
        source1,
        redundancy=redundancy,
        base=base,
        ccs_qmc_samples=ccs_qmc_samples,
    )
    target_tmi = observed["total"].reshape(())
    generator = torch.Generator(device=model.coefficients.device)
    if seed is not None:
        generator.manual_seed(seed)

    coefficients = _random_var_shapes(
        n_null,
        model.n_variables,
        model.order,
        generator=generator,
        dtype=model.coefficients.dtype,
        device=model.coefficients.device,
    )
    covariance = _wishart_identity(
        n_null,
        model.n_variables,
        generator=generator,
        dtype=model.coefficients.dtype,
        device=model.coefficients.device,
    )
    null_model, radius = _match_tmi_by_spectral_radius(
        coefficients,
        covariance,
        target_tmi,
        base=base,
    )
    null_atoms = gaussian_pid_from_var(
        null_model,
        source0,
        source1,
        redundancy=redundancy,
        base=base,
        ccs_qmc_samples=ccs_qmc_samples,
    )
    null_tmi = null_atoms["total"]
    atom_names = ("redundant", "unique_source0", "unique_source1", "synergistic")
    quantile_samples = {name: null_atoms[name] for name in atom_names}
    if str(redundancy).lower() == "mmi":
        pooled_unique = null_atoms["unique_source0"] + null_atoms["unique_source1"]
        quantile_samples["unique_source0"] = pooled_unique
        quantile_samples["unique_source1"] = pooled_unique
    quantiles = {
        name: _empirical_mid_quantile(quantile_samples[name], observed[name])
        for name in atom_names
    }
    return NuMITPIDResult(
        observed=observed,
        quantiles=quantiles,
        null_atoms={name: null_atoms[name] for name in (*atom_names, "total")},
        quantile_samples=quantile_samples,
        target_tmi=target_tmi,
        null_tmi=null_tmi,
        null_spectral_radius=radius,
        n_null=n_null,
        seed=seed,
        redundancy=str(redundancy).lower(),
    )


__all__ = ["NuMITPIDResult", "numit_pid_var", "var_total_mutual_information"]
