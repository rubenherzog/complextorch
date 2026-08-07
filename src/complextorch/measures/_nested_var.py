r"""Nested-VAR primitives used by empirical predictive measures.

Temporal and spectral MVGC require a pair of nested predictive models: a full
model containing target, conditioning variables, and source, and a reduced
model omitting the source. This private module centralizes that construction so
all empirical predictive measures use identical variable ordering, covariance
selection, and log-determinant conventions.

This is measure machinery, not model-order selection: both nested models are
part of the definition of the statistic being evaluated.

References
----------
- Geweke, J. (1982). Measurement of linear dependence and feedback between
  multiple time series. *Journal of the American Statistical Association*.
- Barnett, L. and Seth, A. K. (2014). The MVGC multivariate Granger causality
  toolbox: a new approach to Granger-causal inference. *Journal of
  Neuroscience Methods*.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..linalg import spd_logdet, spd_solve, symmetrise
from ..var import VAR
from .dynamics import cross_spectral_density


def normalise_indices(indices, n_variables: int) -> tuple[int, ...]:
    """Return unique validated variable indices while preserving order.

    Parameters
    ----------
    indices
        One integer or an iterable of integer-like variable indices.
    n_variables
        Total number of observed variables.

    Returns
    -------
    tuple of int
        Unique indices in first-occurrence order.
    """
    if isinstance(indices, int):
        indices = (indices,)
    result = tuple(dict.fromkeys(int(index) for index in indices))
    if not result:
        raise ValueError("at least one variable is required")
    if min(result) < 0 or max(result) >= n_variables:
        raise IndexError("variable index out of range")
    return result


def complement_indices(selected, n_variables: int) -> tuple[int, ...]:
    """Return variables not present in ``selected`` in canonical order."""
    selected_set = set(normalise_indices(selected, n_variables))
    return tuple(
        index for index in range(n_variables) if index not in selected_set
    )


def select_covariance(covariance: torch.Tensor, indices) -> torch.Tensor:
    """Extract a principal covariance block on the final two axes.

    Leading batch or frequency dimensions are preserved.
    """
    index = torch.as_tensor(
        tuple(indices), device=covariance.device, dtype=torch.long
    )
    return covariance.index_select(-2, index).index_select(-1, index)


def conditional_covariance_blocks(
    covariance: torch.Tensor,
    target,
    conditioned,
) -> torch.Tensor:
    r"""Return Gaussian conditional covariance by a Schur complement.

    If the selected covariance is partitioned as

    .. math::

       \Sigma = \begin{bmatrix}
       \Sigma_{AA} & \Sigma_{AB}\\
       \Sigma_{BA} & \Sigma_{BB}
       \end{bmatrix},

    this function returns
    :math:`\Sigma_{A\mid B}=\Sigma_{AA}-\Sigma_{AB}\Sigma_{BB}^{-1}\Sigma_{BA}`.
    The solve uses the shared SPD routine rather than an explicit inverse.
    """
    target = tuple(target)
    conditioned = tuple(conditioned)
    covariance_aa = select_covariance(covariance, target)
    if not conditioned:
        return covariance_aa
    index_a = torch.as_tensor(target, device=covariance.device)
    index_b = torch.as_tensor(conditioned, device=covariance.device)
    covariance_ab = covariance.index_select(-2, index_a).index_select(
        -1, index_b
    )
    covariance_bb = select_covariance(covariance, conditioned)
    conditional = covariance_aa - covariance_ab @ spd_solve(
        covariance_bb, covariance_ab.transpose(-1, -2)
    )
    return symmetrise(conditional)


def logdet_ratio(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    *,
    base: float = math.e,
) -> torch.Tensor:
    r"""Return :math:`\log(|A|/|B|)` in the requested logarithm base.

    ``numerator`` and ``denominator`` must be positive-definite covariance
    matrices, optionally with matching leading batch dimensions. Log
    determinants are evaluated through the shared SPD factorization for
    numerical stability.
    """
    return (
        spd_logdet(numerator) - spd_logdet(denominator)
    ) / math.log(base)


@dataclass(frozen=True)
class NestedVARModels:
    """Full and reduced VAR fits with their canonical variable mappings."""

    full: VAR
    reduced: VAR
    full_variables: tuple[int, ...]
    reduced_variables: tuple[int, ...]
    target_positions_full: tuple[int, ...]
    target_positions_reduced: tuple[int, ...]


def fit_nested_var_models(
    observations: torch.Tensor,
    order: int,
    target,
    source,
    conditional=None,
    **var_kwargs,
) -> NestedVARModels:
    """Fit consistent full and reduced VAR models for predictive measures.

    The reduced ordering is ``target + conditional`` and the full ordering is
    ``target + conditional + source``. If ``conditional`` is ``None``, all
    variables outside target and source are used as conditioning variables;
    an empty iterable requests an unconditional source-to-target comparison.

    Both fits receive the same VAR settings. By default, batched trajectories
    are pooled through :class:`VAR` sufficient statistics, which preserves
    trajectory boundaries rather than concatenating raw time series.
    """
    values = torch.as_tensor(observations)
    n_variables = values.shape[-1]
    target = normalise_indices(target, n_variables)
    source = normalise_indices(source, n_variables)
    if set(target) & set(source):
        raise ValueError("source and target must be disjoint")

    if conditional is None:
        conditional = tuple(
            index
            for index in range(n_variables)
            if index not in target and index not in source
        )
    elif len(tuple(conditional)) == 0:
        conditional = ()
    else:
        conditional = tuple(
            index
            for index in normalise_indices(conditional, n_variables)
            if index not in target and index not in source
        )

    reduced_variables = target + conditional
    full_variables = reduced_variables + source
    defaults = {
        "mode": "pooled",
        "fit_intercept": True,
        "stability": "ignore",
    }
    defaults.update(var_kwargs)
    full = VAR(order=order, **defaults).fit(values[..., full_variables])
    reduced = VAR(order=order, **defaults).fit(values[..., reduced_variables])
    target_positions = tuple(range(len(target)))
    return NestedVARModels(
        full,
        reduced,
        full_variables,
        reduced_variables,
        target_positions,
        target_positions,
    )


def residual_target_covariance(
    model: VAR,
    target_positions,
) -> torch.Tensor:
    """Return the target block of a fitted VAR innovation covariance."""
    return select_covariance(model.noise_covariance_, target_positions)


def conditional_spectrum(
    spectrum: torch.Tensor,
    target,
    conditioned=(),
) -> torch.Tensor:
    r"""Return the frequency-wise Schur-complement target spectrum.

    The final two tensor axes are treated as spectral covariance axes; all
    leading batch and frequency axes are preserved. ``torch.linalg.solve`` is
    used instead of forming :math:`S_{BB}^{-1}` explicitly.
    """
    target = tuple(target)
    conditioned = tuple(conditioned)
    spectrum_aa = select_covariance(spectrum, target)
    if not conditioned:
        return spectrum_aa
    index_a = torch.as_tensor(target, device=spectrum.device)
    index_b = torch.as_tensor(conditioned, device=spectrum.device)
    spectrum_ab = spectrum.index_select(-2, index_a).index_select(-1, index_b)
    spectrum_bb = select_covariance(spectrum, conditioned)
    conditional = spectrum_aa - spectrum_ab @ torch.linalg.solve(
        spectrum_bb, spectrum_ab.transpose(-1, -2)
    )
    return symmetrise(conditional)


def var_model_spectrum(
    model: VAR,
    frequencies: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the cross-spectral density of a fitted VAR model.

    Parameters
    ----------
    model
        Fitted :class:`VAR` estimator.
    frequencies
        One-dimensional frequency grid in normalized cycles per sample.

    Returns
    -------
    torch.Tensor
        Cross-spectral density returned by the canonical ``VARSystem`` path.
    """
    return cross_spectral_density(model.to_var_system(), frequencies)
