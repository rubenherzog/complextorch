"""Shared full/reduced-model utilities for predictive dependence measures."""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from ..linalg import spd_logdet, spd_solve, symmetrise
from ..var import VAR
from .dynamics import cross_spectral_density


def normalise_indices(indices, n_variables: int) -> tuple[int, ...]:
    if isinstance(indices, int):
        indices = (indices,)
    result = tuple(dict.fromkeys(int(i) for i in indices))
    if not result:
        raise ValueError("at least one variable is required")
    if min(result) < 0 or max(result) >= n_variables:
        raise IndexError("variable index out of range")
    return result


def complement_indices(selected, n_variables: int) -> tuple[int, ...]:
    selected = set(normalise_indices(selected, n_variables))
    return tuple(i for i in range(n_variables) if i not in selected)


def select_covariance(covariance: torch.Tensor, indices) -> torch.Tensor:
    idx = torch.as_tensor(tuple(indices), device=covariance.device, dtype=torch.long)
    return covariance.index_select(-2, idx).index_select(-1, idx)


def conditional_covariance_blocks(covariance: torch.Tensor, target, conditioned) -> torch.Tensor:
    """Cov(target | conditioned) using the common SPD solver."""
    target, conditioned = tuple(target), tuple(conditioned)
    aa = select_covariance(covariance, target)
    if not conditioned:
        return aa
    idx_a = torch.as_tensor(target, device=covariance.device)
    idx_b = torch.as_tensor(conditioned, device=covariance.device)
    ab = covariance.index_select(-2, idx_a).index_select(-1, idx_b)
    bb = select_covariance(covariance, conditioned)
    return symmetrise(aa - ab @ spd_solve(bb, ab.transpose(-1, -2)))


def logdet_ratio(numerator: torch.Tensor, denominator: torch.Tensor, *, base: float = math.e) -> torch.Tensor:
    return (spd_logdet(numerator) - spd_logdet(denominator)) / math.log(base)


@dataclass(frozen=True)
class NestedVARModels:
    full: VAR
    reduced: VAR
    full_variables: tuple[int, ...]
    reduced_variables: tuple[int, ...]
    target_positions_full: tuple[int, ...]
    target_positions_reduced: tuple[int, ...]


def fit_nested_var_models(observations: torch.Tensor, order: int, target, source, conditional=(), **var_kwargs) -> NestedVARModels:
    """Fit consistent full/reduced VARs once for all predictive measures."""
    x = torch.as_tensor(observations)
    n = x.shape[-1]
    target = normalise_indices(target, n)
    source = normalise_indices(source, n)
    conditional = tuple(i for i in normalise_indices(conditional, n) if i not in target and i not in source) if conditional else ()
    if set(target) & set(source):
        raise ValueError("source and target must be disjoint")
    reduced_vars = target + conditional
    full_vars = reduced_vars + source
    defaults = dict(mode="pooled", fit_intercept=True, stability="ignore")
    defaults.update(var_kwargs)
    full = VAR(order=order, **defaults).fit(x[..., full_vars])
    reduced = VAR(order=order, **defaults).fit(x[..., reduced_vars])
    positions = tuple(range(len(target)))
    return NestedVARModels(full, reduced, full_vars, reduced_vars, positions, positions)


def residual_target_covariance(model: VAR, target_positions) -> torch.Tensor:
    return select_covariance(model.noise_covariance_, target_positions)


def conditional_spectrum(spectrum: torch.Tensor, target, conditioned=()) -> torch.Tensor:
    """Frequency-wise Schur-complement conditional spectrum."""
    target, conditioned = tuple(target), tuple(conditioned)
    aa = select_covariance(spectrum, target)
    if not conditioned:
        return aa
    idx_a = torch.as_tensor(target, device=spectrum.device)
    idx_b = torch.as_tensor(conditioned, device=spectrum.device)
    ab = spectrum.index_select(-2, idx_a).index_select(-1, idx_b)
    bb = select_covariance(spectrum, conditioned)
    return symmetrise(aa - ab @ torch.linalg.solve(bb, ab.transpose(-1, -2)))


def var_model_spectrum(model: VAR, frequencies: torch.Tensor) -> torch.Tensor:
    return cross_spectral_density(model.to_var_system(), frequencies)
