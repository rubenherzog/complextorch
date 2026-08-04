"""Temporal and spectral Granger-causality utilities."""
from __future__ import annotations
import math
import torch
from ..var import VAR
from ..representations import VARSystem
from .dynamics import transfer_function, cross_spectral_density


def temporal_mvgc(observations, order: int, source: int, target: int, *, conditional=None, base: float = math.e):
    x = torch.as_tensor(observations)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    keep = [target] + ([] if conditional is None else [i for i in conditional if i not in (source, target)])
    full_keep = keep + [source]
    full = VAR(order=order, mode="pooled", fit_intercept=True, stability="ignore").fit(x[..., full_keep])
    reduced = VAR(order=order, mode="pooled", fit_intercept=True, stability="ignore").fit(x[..., keep])
    full_variance = full.noise_covariance_[0, 0, 0]
    reduced_variance = reduced.noise_covariance_[0, 0, 0]
    return torch.log(reduced_variance / full_variance) / math.log(base)


def pairwise_spectral_gc(system: VARSystem, source: int, target: int, frequencies, *, base: float = math.e):
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
