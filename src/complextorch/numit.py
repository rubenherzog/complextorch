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
    # MVGC2 ``var_rand`` starts from iid standard-normal lag matrices. Its
    # intermediate random target radius cancels algebraically when ``specnorm``
    # is applied again to the NuMIT-optimised radius, so keeping the raw shapes
    # is exactly equivalent and avoids an unnecessary eigendecomposition.
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
    """Return dtype-aware open bounds for stable spectral-radius matching.

    NuMIT's MATLAB reference optimises an unconstrained variable transformed by
    a sigmoid, so the theoretical radius range is the full open interval
    ``(0, 1)``. A fixed ``1e-5`` margin truncates high-TMI systems unnecessarily
    in float64. Here the margin is tied to floating-point resolution instead.

    ``64 * eps`` leaves enough distance from the unit root for companion
    eigenvalue and Lyapunov stability checks while reaching approximately
    ``1 - 1.4e-14`` in float64 and ``1 - 7.6e-6`` in float32.
    """
    if not dtype.is_floating_point:
        raise TypeError("NuMIT spectral-radius matching requires floating-point dtype")
    margin = 64.0 * torch.finfo(dtype).eps
    low = torch.tensor(margin, dtype=dtype, device=device)
    high = torch.tensor(1.0 - margin, dtype=dtype, device=device)
    return low, high


def _match_tmi_by_spectral_radius(
    coefficients: torch.Tensor,
    covariance: torch.Tensor,
    target_tmi: torch.Tensor,
    *,
    base: float,
    iterations: int = 64,
) -> tuple[VARSystem, torch.Tensor]:
    """Match each random VAR to one target TMI using batched bisection in rho.

    The search covers the widest numerically safe subset of the reference
    interval ``(0, 1)`` for the input dtype. Consequently, float64 nulls can
    approach the unit-root boundary much more closely than float32 nulls.
    """
    batch = coefficients.shape[0]
    target = torch.as_tensor(target_tmi, dtype=coefficients.dtype, device=coefficients.device)
    if target.ndim == 0:
        target = target.expand(batch)
    if target.shape != (batch,):
        raise ValueError("target_tmi must be scalar or have shape (batch,)")
    if bool((target < 0).any()):
        raise ValueError("target_tmi must be nonnegative")

    lower_bound, upper_bound = _numerical_radius_bounds(
        coefficients.dtype,
        device=coefficients.device,
    )
    low = torch.full_like(target, lower_bound.item())
    high = torch.full_like(target, upper_bound.item())

    low_model = build_var_system(_var_decay_to_radius(coefficients, low), covariance)
    low_tmi = var_total_mutual_information(low_model, base=base)
    high_model = build_var_system(_var_decay_to_radius(coefficients, high), covariance)
    high_tmi = var_total_mutual_information(high_model, base=base)

    below_range = target < low_tmi
    above_range = target > high_tmi
    if bool((below_range | above_range).any()):
        bad = below_range | above_range
        raise RuntimeError(
            "target TMI lies outside the numerically reachable range for "
            f"{int(bad.sum().item())} sampled null VAR(s); "
            f"dtype={coefficients.dtype}, radius range="
            f"({float(lower_bound):.3e}, {float(upper_bound):.17g}), "
            f"requested TMI range=({float(target[bad].min()):.6g}, "
            f"{float(target[bad].max()):.6g})"
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
    """Return the NuMIT/``CompQuantile`` empirical mid-quantile.

    The reference implementation counts values strictly below ``value-atol``
    plus half the values equal within ``atol``. Non-finite null samples are
    excluded from the denominator.
    """
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
    """NuMIT-normalise Gaussian VAR PID atoms against TMI-matched null VARs.

    The two source groups must form a partition of all observed channels. Each
    source comprises the complete VAR history of those channels and the target
    is the joint future state, matching Liardi et al. (2025).
    """
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
        # Reference NuMIT pools the two complementary MMI unique atoms. Exactly
        # one is zero for each null system, so using either raw distribution
        # would create a degenerate mass at zero. Both observed unique atoms are
        # ranked against U0 + U1 instead (NuMIT_PID.m, commit pinned above).
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
