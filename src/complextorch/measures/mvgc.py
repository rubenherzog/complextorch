"""Temporal and spectral multivariate Granger causality.

Two explicit routes are provided:
- regression MVGC, based on separately fitted nested VAR models;
- exact state-space MVGC, based on one full innovations model and DARE-derived marginals.

Notes
-----
Conditional time-domain Granger causality is the log ratio of reduced and full
innovation generalised variances,

.. math::

   F_{Y	o X\mid Z}=\log
rac{\det\Sigma^{R}_{XX}}
                                {\det\Sigma_{XX}}.

Spectral GC is computed from innovations-form transfer functions and integrates
to the time-domain value under the Geweke decomposition.

References
----------
- Geweke, J. (1982). Measurement of linear dependence and feedback between
  multiple time series.
- Barnett, L. and Seth, A. K. (2014), MVGC toolbox paper.
- Barnett, L. and Seth, A. K. (2015), state-space Granger causality.
- MVGC repository: https://github.com/lcbarnett/MVGC1

Notes
-----
Conditional time-domain Granger causality is the log ratio of reduced and full
innovation generalised variances,

.. math::

   F_{Y	o X\mid Z}=\log
rac{\det\Sigma^{R}_{XX}}
                                {\det\Sigma_{XX}}.

Spectral GC is computed from innovations-form transfer functions and integrates
to the time-domain value under the Geweke decomposition.

References
----------
- Geweke, J. (1982). Measurement of linear dependence and feedback between
  multiple time series.
- Barnett, L. and Seth, A. K. (2014), MVGC toolbox paper.
- Barnett, L. and Seth, A. K. (2015), state-space Granger causality.
- MVGC repository: https://github.com/lcbarnett/MVGC1
"""
from __future__ import annotations
import math
import torch
from ..representations import VARSystem
from ..control import InnovationsStateSpace, var_to_innovations_state_space, reduce_innovations_state_space, innovations_transfer_function
from ..linalg import spd_logdet, spd_solve
from ._model_comparison import fit_nested_var_models, residual_target_covariance, logdet_ratio, conditional_spectrum, var_model_spectrum, normalise_indices
from .dynamics import transfer_function, cross_spectral_density


def temporal_mvgc(observations: torch.Tensor, order: int, source, target, *, conditional=(), base: float = math.e, **var_kwargs) -> torch.Tensor:
    """Regression conditional group GC using separately fitted nested VARs.
        
        Compute conditional time-domain multivariate Granger causality.
        
        .. math:: F_{Y	o X\mid Z}=\log(\det\Sigma^R_{XX}/\det\Sigma_{XX}).
        
        References
        ----------
        Geweke (1982); Barnett and Seth (2014, 2015).
    
    Compute conditional time-domain multivariate Granger causality.
    
    .. math:: F_{Y	o X\mid Z}=\log(\det\Sigma^R_{XX}/\det\Sigma_{XX}).
    
    References
    ----------
    Geweke (1982); Barnett and Seth (2014, 2015).
    """
    models = fit_nested_var_models(observations, order, target, source, conditional, **var_kwargs)
    full_cov = residual_target_covariance(models.full, models.target_positions_full)
    reduced_cov = residual_target_covariance(models.reduced, models.target_positions_reduced)
    return logdet_ratio(reduced_cov, full_cov, base=base)


def spectral_mvgc(observations: torch.Tensor, order: int, source, target, frequencies: torch.Tensor, *, conditional=(), base: float = math.e, **var_kwargs) -> torch.Tensor:
    """Regression spectral GC from the same separately fitted full/reduced VARs.
        
        Compute conditional spectral multivariate Granger causality.
        
        The frequency-resolved decomposition is obtained from innovations-form transfer
        functions and integrates to temporal GC.
        
        References
        ----------
        Geweke (1982); Barnett and Seth (2014, 2015).
    
    Compute conditional spectral multivariate Granger causality.
    
    The frequency-resolved decomposition is obtained from innovations-form transfer
    functions and integrates to temporal GC.
    
    References
    ----------
    Geweke (1982); Barnett and Seth (2014, 2015).
    """
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


def _as_innovations(system: VARSystem | InnovationsStateSpace) -> InnovationsStateSpace:
    """ as innovations.
    
    Parameters
    ----------
    system
        Input controlling ``_as_innovations``.
    
    Returns
    -------
    object
        Result described by the function name and annotated return type.
    
    Notes
    -----
    Tensor batch dimensions are preserved unless the public API explicitly
    documents a squeeze operation. Numerical validation is performed by the
    module before the core calculation.
    """
    return var_to_innovations_state_space(system) if isinstance(system, VARSystem) else system


def _normalise_partition(system: InnovationsStateSpace, target, source, conditional=()):
    """ normalise partition.
    
    Parameters
    ----------
    system
        Input controlling ``_normalise_partition``.
    target
        Input controlling ``_normalise_partition``.
    source
        Input controlling ``_normalise_partition``.
    conditional
        Input controlling ``_normalise_partition``.
    
    Returns
    -------
    object
        Result described by the function name and annotated return type.
    
    Notes
    -----
    Tensor batch dimensions are preserved unless the public API explicitly
    documents a squeeze operation. Numerical validation is performed by the
    module before the core calculation.
    """
    n_variables = system.observation.shape[-2]
    target = normalise_indices(target, n_variables)
    source = normalise_indices(source, n_variables)
    conditional = normalise_indices(conditional, n_variables) if conditional else ()
    if set(target) & set(source) or set(target) & set(conditional) or set(source) & set(conditional):
        raise ValueError("target, source and conditional sets must be pairwise disjoint")
    return target, source, conditional


def _hermitian(matrix: torch.Tensor) -> torch.Tensor:
    """ hermitian.
    
    Parameters
    ----------
    matrix
        Input controlling ``_hermitian``.
    
    Returns
    -------
    object
        Result described by the function name and annotated return type.
    
    Notes
    -----
    Tensor batch dimensions are preserved unless the public API explicitly
    documents a squeeze operation. Numerical validation is performed by the
    module before the core calculation.
    """
    return 0.5 * (matrix + matrix.conj().transpose(-1, -2))


def _joint_state_space_spectral_gc(
    system: InnovationsStateSpace,
    target,
    drivers,
    frequencies: torch.Tensor,
    *,
    base: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Joint drivers→target spectral GC and target innovations covariance."""
    order = tuple(target) + tuple(drivers)
    marginal = reduce_innovations_state_space(system, order)
    transfer = innovations_transfer_function(marginal, frequencies)
    covariance = marginal.innovation_covariance
    single = transfer.ndim == 3
    if single:
        transfer = transfer.unsqueeze(0)
        covariance = covariance.unsqueeze(0)
    n_target = len(target)
    n_total = n_target + len(drivers)
    target_covariance = covariance[..., :n_target, :n_target]
    source_target_covariance = covariance[..., n_target:, :n_target]
    identity_target = torch.eye(n_target, dtype=covariance.dtype, device=covariance.device).expand(*covariance.shape[:-2], n_target, n_target)
    regression = source_target_covariance @ spd_solve(target_covariance, identity_target)
    transform = torch.eye(n_total, dtype=covariance.dtype, device=covariance.device).expand(*covariance.shape[:-2], n_total, n_total).clone()
    transform[..., n_target:, :n_target] = regression
    normalized_transfer = transfer @ transform[:, None].to(transfer.dtype)
    spectrum = transfer @ covariance[:, None].to(transfer.dtype) @ transfer.conj().transpose(-1, -2)
    intrinsic_transfer = normalized_transfer[..., :n_target, :n_target]
    intrinsic = intrinsic_transfer @ target_covariance[:, None].to(transfer.dtype) @ intrinsic_transfer.conj().transpose(-1, -2)
    total_target = _hermitian(spectrum[..., :n_target, :n_target])
    intrinsic = _hermitian(intrinsic)
    value = (torch.linalg.slogdet(total_target).logabsdet - torch.linalg.slogdet(intrinsic).logabsdet) / math.log(base)
    return (value[0], target_covariance[0]) if single else (value, target_covariance)


def state_space_temporal_mvgc(
    system: VARSystem | InnovationsStateSpace,
    source,
    target,
    *,
    conditional=(),
    base: float = math.e,
) -> torch.Tensor:
    """Exact conditional GC from one full model and DARE-derived marginals."""
    innovations = _as_innovations(system)
    target, source, conditional = _normalise_partition(innovations, target, source, conditional)
    full = reduce_innovations_state_space(innovations, target + conditional + source)
    reduced = reduce_innovations_state_space(innovations, target + conditional)
    n_target = len(target)
    full_covariance = full.innovation_covariance[..., :n_target, :n_target]
    reduced_covariance = reduced.innovation_covariance[..., :n_target, :n_target]
    # Evaluate log-determinants through an SPD-aware factorisation for numerical stability.
    return (spd_logdet(reduced_covariance) - spd_logdet(full_covariance)) / math.log(base)


def state_space_spectral_mvgc(
    system: VARSystem | InnovationsStateSpace,
    source,
    target,
    frequencies: torch.Tensor,
    *,
    conditional=(),
    base: float = math.e,
) -> torch.Tensor:
    """Exact conditional spectral GC using state-space marginalisation.

    Conditional source→target|conditional is computed as
    joint(source,conditional)→target minus conditional→target. Both terms are
    derived from the same full innovations model through generalized DAREs.
    """
    innovations = _as_innovations(system)
    target, source, conditional = _normalise_partition(innovations, target, source, conditional)
    joint, _ = _joint_state_space_spectral_gc(innovations, target, source + conditional, frequencies, base=base)
    if not conditional:
        return joint
    conditioning, _ = _joint_state_space_spectral_gc(innovations, target, conditional, frequencies, base=base)
    return joint - conditioning


def integrate_spectral_mvgc(values: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
    """Integrate one-sided GC on normalized frequencies [0,.5] to time GC.
        
        Integrate one-sided spectral GC to its time-domain value.
        
        For normalised frequencies :math:`f\in[0,1/2]`, the implementation evaluates
        :math:`2\int_0^{1/2} f_{Y	o X}(
        u)\,d
        u`.
        
        References
        ----------
        Geweke (1982); Barnett and Seth (2014).
    
    Integrate one-sided spectral GC to its time-domain value.
    
    For normalised frequencies :math:`f\in[0,1/2]`, the implementation evaluates
    :math:`2\int_0^{1/2} f_{Y	o X}(
    u)\,d
    u`.
    
    References
    ----------
    Geweke (1982); Barnett and Seth (2014).
    """
    frequencies = torch.as_tensor(frequencies, dtype=values.real.dtype, device=values.device)
    if frequencies.ndim != 1 or frequencies.numel() != values.shape[-1]:
        raise ValueError("frequencies must match the last dimension of values")
    # One-sided spectral integration recovers the corresponding time-domain quantity.
    return 2.0 * torch.trapezoid(values, frequencies, dim=-1)


def pairwise_spectral_gc(system: VARSystem, source: int, target: int, frequencies: torch.Tensor, *, base: float = math.e) -> torch.Tensor:
    """Backward-compatible bivariate Geweke spectral GC."""
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