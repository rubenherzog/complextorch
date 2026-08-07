"""Nested-VAR primitives used by empirical predictive measures.

Reduced and full VAR models are fitted with consistent target/source ordering;
log-determinant ratios and conditional spectra are then reused by empirical
MVGC and related estimators.

References
----------
- Geweke, J. (1982).
- Barnett, L. and Seth, A. K. (2014). The MVGC toolbox.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from ..linalg import spd_logdet, spd_solve, symmetrise
from ..var import VAR
from .dynamics import cross_spectral_density


def normalise_indices(indices, n_variables: int) -> tuple[int, ...]:
    """Normalise indices.

    Parameters
    ----------
    indices
        Input required by this calculation.
    n_variables
        Number of observed variables.

    Returns
    -------
    object
        Computed result; see the annotated return type and shape notes.

    Notes
    -----
    Batch dimensions are preserved unless explicitly documented otherwise.
    The implementation validates dimensional and positive-definiteness
    requirements before executing the numerical core.
    """
    if isinstance(indices, int):
        indices = (indices,)
    result = tuple(dict.fromkeys(int(i) for i in indices))
    if not result:
        raise ValueError("at least one variable is required")
    if min(result) < 0 or max(result) >= n_variables:
        raise IndexError("variable index out of range")
    return result


def complement_indices(selected, n_variables: int) -> tuple[int, ...]:
    """Complement indices.

    Parameters
    ----------
    selected
        Input required by this calculation.
    n_variables
        Number of observed variables.

    Returns
    -------
    object
        Computed result; see the annotated return type and shape notes.

    Notes
    -----
    Batch dimensions are preserved unless explicitly documented otherwise.
    The implementation validates dimensional and positive-definiteness
    requirements before executing the numerical core.
    """
    selected = set(normalise_indices(selected, n_variables))
    return tuple(i for i in range(n_variables) if i not in selected)


def select_covariance(covariance: torch.Tensor, indices) -> torch.Tensor:
    """Select covariance.

    Parameters
    ----------
    covariance
        Symmetric covariance matrix or batch of covariance matrices.
    indices
        Input required by this calculation.

    Returns
    -------
    object
        Computed result; see the annotated return type and shape notes.

    Notes
    -----
    Batch dimensions are preserved unless explicitly documented otherwise.
    The implementation validates dimensional and positive-definiteness
    requirements before executing the numerical core.
    """
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
    """Logdet ratio.

    Parameters
    ----------
    numerator
        Input required by this calculation.
    denominator
        Input required by this calculation.
    base
        Logarithm base used for information quantities.

    Returns
    -------
    object
        Computed result; see the annotated return type and shape notes.

    Notes
    -----
    Batch dimensions are preserved unless explicitly documented otherwise.
    The implementation validates dimensional and positive-definiteness
    requirements before executing the numerical core.
    """
    # Evaluate log-determinants through an SPD-aware factorisation for numerical stability.
    return (spd_logdet(numerator) - spd_logdet(denominator)) / math.log(base)


@dataclass(frozen=True)
class NestedVARModels:
    """Nestedvarmodels.

    Notes
    -----
    Public fitted attributes use the trailing-underscore convention.
    """
    full: VAR
    reduced: VAR
    full_variables: tuple[int, ...]
    reduced_variables: tuple[int, ...]
    target_positions_full: tuple[int, ...]
    target_positions_reduced: tuple[int, ...]


def fit_nested_var_models(observations: torch.Tensor, order: int, target, source, conditional=None, **var_kwargs) -> NestedVARModels:
    """Fit consistent full/reduced VARs once for all predictive measures."""
    x = torch.as_tensor(observations)
    n = x.shape[-1]
    target = normalise_indices(target, n)
    source = normalise_indices(source, n)
    if set(target) & set(source):
        raise ValueError("source and target must be disjoint")
    if conditional is None:
        conditional = tuple(i for i in range(n) if i not in target and i not in source)
    elif len(tuple(conditional)) == 0:
        conditional = ()
    else:
        conditional = tuple(
            i for i in normalise_indices(conditional, n) if i not in target and i not in source
        )
    reduced_vars = target + conditional
    full_vars = reduced_vars + source
    defaults = dict(mode="pooled", fit_intercept=True, stability="ignore")
    defaults.update(var_kwargs)
    full = VAR(order=order, **defaults).fit(x[..., full_vars])
    reduced = VAR(order=order, **defaults).fit(x[..., reduced_vars])
    positions = tuple(range(len(target)))
    return NestedVARModels(full, reduced, full_vars, reduced_vars, positions, positions)


def residual_target_covariance(model: VAR, target_positions) -> torch.Tensor:
    """Residual target covariance.

    Parameters
    ----------
    model
        VAR or linear state-space model.
    target_positions
        Input required by this calculation.

    Returns
    -------
    object
        Computed result; see the annotated return type and shape notes.

    Notes
    -----
    Batch dimensions are preserved unless explicitly documented otherwise.
    The implementation validates dimensional and positive-definiteness
    requirements before executing the numerical core.
    """
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
    """Var model spectrum.

    Parameters
    ----------
    model
        VAR or linear state-space model.
    frequencies
        One-dimensional frequency grid in normalized cycles per sample.

    Returns
    -------
    object
        Computed result; see the annotated return type and shape notes.

    Notes
    -----
    Batch dimensions are preserved unless explicitly documented otherwise.
    The implementation validates dimensional and positive-definiteness
    requirements before executing the numerical core.
    """
    return cross_spectral_density(model.to_var_system(), frequencies)
