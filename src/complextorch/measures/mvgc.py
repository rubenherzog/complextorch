"""Temporal and spectral multivariate Granger causality."""
from __future__ import annotations
import math
import torch
from ..representations import VARSystem
from ._model_comparison import fit_nested_var_models, residual_target_covariance, logdet_ratio, conditional_spectrum, var_model_spectrum, normalise_indices
from .dynamics import transfer_function, cross_spectral_density


def temporal_mvgc(observations: torch.Tensor, order: int, source, target, *, conditional=(), base: float = math.e, **var_kwargs) -> torch.Tensor:
    """Conditional group time-domain GC using shared nested VAR models."""
    models = fit_nested_var_models(observations, order, target, source, conditional, **var_kwargs)
    full_cov = residual_target_covariance(models.full, models.target_positions_full)
    reduced_cov = residual_target_covariance(models.reduced, models.target_positions_reduced)
    return logdet_ratio(reduced_cov, full_cov, base=base)


def spectral_mvgc(observations: torch.Tensor, order: int, source, target, frequencies: torch.Tensor, *, conditional=(), base: float = math.e, **var_kwargs) -> torch.Tensor:
    """Group spectral GC from the same full/reduced models at every frequency."""
    models = fit_nested_var_models(observations, order, target, source, conditional, **var_kwargs)
    full_spectrum = var_model_spectrum(models.full, frequencies)
    reduced_spectrum = var_model_spectrum(models.reduced, frequencies)
    n_target = len(models.target_positions_full)
    n_source = len(normalise_indices(source, observations.shape[-1]))
    conditioned_full = tuple(range(n_target, len(models.full_variables) - n_source))
    conditioned_reduced = tuple(range(n_target, len(models.reduced_variables)))
    full_target = conditional_spectrum(full_spectrum, range(n_target), conditioned_full)
    reduced_target = conditional_spectrum(reduced_spectrum, range(n_target), conditioned_reduced)
    return logdet_ratio(reduced_target, full_target, base=base)


def pairwise_spectral_gc(system: VARSystem, source: int, target: int, frequencies: torch.Tensor, *, base: float = math.e) -> torch.Tensor:
    """Bivariate Geweke spectral GC with innovation normalisation."""
    if system.n_variables != 2:
        raise ValueError("pairwise_spectral_gc expects a bivariate VARSystem")
    transfer = transfer_function(system, frequencies)
    spectrum = cross_spectral_density(system, frequencies)
    noise = system.innovation_covariance
    if source == 0 and target == 1:
        permutation = torch.tensor([1, 0], device=noise.device)
        noise = noise.index_select(-2, permutation).index_select(-1, permutation)
        transfer = transfer.index_select(-2, permutation).index_select(-1, permutation)
        spectrum = spectrum.index_select(-2, permutation).index_select(-1, permutation)
    elif not (source == 1 and target == 0):
        raise ValueError("source and target must differ and be 0/1")
    sigma_11 = noise[..., 0, 0]
    sigma_12 = noise[..., 0, 1]
    normalized_h11 = transfer[..., 0, 0] + (sigma_12 / sigma_11)[..., None] * transfer[..., 0, 1]
    intrinsic = normalized_h11.abs().square() * sigma_11[..., None]
    total = spectrum[..., 0, 0].real.clamp_min(torch.finfo(spectrum.real.dtype).eps)
    return torch.log(total / intrinsic.clamp_min(torch.finfo(total.dtype).eps)) / math.log(base)
