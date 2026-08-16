"""Temporal and spectral multivariate Granger causality.

Conditional time-domain GC is

.. math::

   F_{Y\to X\mid Z}
   =\log\frac{\det\Sigma^{R}_{XX}}{\det\Sigma_{XX}},

where the reduced covariance excludes the source history. Spectral GC uses
innovations-form transfer functions and integrates to the time-domain value.

References
----------
- Geweke, J. (1982). Measurement of linear dependence and feedback.
- Barnett, L. and Seth, A. K. (2014). The MVGC toolbox.
- Barnett, L. and Seth, A. K. (2015). State-space Granger causality.
- MVGC: https://github.com/lcbarnett/MVGC1
"""
from __future__ import annotations
import math
import torch
from ..representations import VARSystem
from ..control import InnovationsStateSpace, var_to_innovations_state_space, reduce_innovations_state_space, innovations_transfer_function
from ..linalg import spd_logdet, spd_solve
from ._nested_var import fit_nested_var_models, residual_target_covariance, logdet_ratio, conditional_spectrum, var_model_spectrum, normalise_indices
from .dynamics import transfer_function, cross_spectral_density


def temporal_mvgc(observations: torch.Tensor, order: int, source, target, *, conditional=None, base: float = math.e, **var_kwargs) -> torch.Tensor:
    """Compute conditional time-domain multivariate Granger causality.

    .. math::

       F_{Y\to X\mid Z}
       =\log\frac{\det\Sigma^{R}_{XX}}{\det\Sigma_{XX}}.

    References
    ----------
    - Geweke (1982); Barnett and Seth (2014, 2015).
    """
    # Compare full and reduced innovation covariance volumes to obtain Geweke time-domain Granger causality.
    models = fit_nested_var_models(observations, order, target, source, conditional, **var_kwargs)
    full_cov = residual_target_covariance(models.full, models.target_positions_full)
    reduced_cov = residual_target_covariance(models.reduced, models.target_positions_reduced)
    return logdet_ratio(reduced_cov, full_cov, base=base)


def spectral_mvgc(observations: torch.Tensor, order: int, source, target, frequencies: torch.Tensor, *, conditional=None, base: float = math.e, **var_kwargs) -> torch.Tensor:
    """Compute conditional spectral multivariate Granger causality.

    The frequency-resolved decomposition is obtained from innovations-form transfer
    functions and integrates to temporal GC.

    References
    ----------
    - Geweke (1982); Barnett and Seth (2014, 2015).
    """
    # Decompose the predictive covariance ratio over frequency using the model transfer function and spectrum.
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
    """As innovations.

    Parameters
    ----------
    system
        Canonical VAR or state-space system.

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
    return var_to_innovations_state_space(system) if isinstance(system, VARSystem) else system


def _normalise_partition(system: InnovationsStateSpace, target, source, conditional=None):
    """Normalise partition.

    Parameters
    ----------
    system
        Canonical VAR or state-space system.
    target
        Indices of target variables.
    source
        Indices of source variables.
    conditional
        Indices conditioned on in addition to source and target.

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
    n_variables = system.observation.shape[-2]
    target = normalise_indices(target, n_variables)
    source = normalise_indices(source, n_variables)
    if set(target) & set(source):
        raise ValueError("target and source sets must be disjoint")
    if conditional is None:
        conditional = tuple(
            index for index in range(n_variables) if index not in target and index not in source
        )
    elif len(tuple(conditional)) == 0:
        conditional = ()
    else:
        conditional = normalise_indices(conditional, n_variables)
    if set(target) & set(conditional) or set(source) & set(conditional):
        raise ValueError("target, source and conditional sets must be pairwise disjoint")
    return target, source, conditional


def _hermitian(matrix: torch.Tensor) -> torch.Tensor:
    """Hermitian.

    Parameters
    ----------
    matrix
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
    conditional=None,
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


def pairwise_temporal_mvgc(
    system: VARSystem | InnovationsStateSpace,
    *,
    base: float = math.e,
) -> torch.Tensor:
    r"""Return all conditional singleton MVGC values with one reduction per source.

    For :math:`F_{i\to j\mid X\setminus\{i,j\}}`, removing source ``i``
    defines the same reduced model for every target ``j != i``. This function
    therefore performs only ``n_variables`` reduced-innovations DARE solves,
    rather than one reduction per ordered pair, while preserving the exact
    conditional-MVGC definition.
    """
    innovations = _as_innovations(system)
    n_variables = innovations.observation.shape[-2]
    covariance = innovations.innovation_covariance
    result = torch.zeros(
        (*covariance.shape[:-2], n_variables, n_variables),
        dtype=covariance.dtype,
        device=covariance.device,
    )
    if n_variables < 2:
        return result
    full_variances = torch.diagonal(covariance, dim1=-2, dim2=-1)
    for source in range(n_variables):
        keep = tuple(index for index in range(n_variables) if index != source)
        reduced = reduce_innovations_state_space(innovations, keep)
        reduced_variances = torch.diagonal(
            reduced.innovation_covariance, dim1=-2, dim2=-1
        )
        targets = torch.as_tensor(keep, dtype=torch.long, device=covariance.device)
        values = (
            torch.log(reduced_variances)
            - torch.log(full_variances.index_select(-1, targets))
        ) / math.log(base)
        result[..., source, targets] = values
    return result


def maximum_temporal_mvgc(
    system: VARSystem | InnovationsStateSpace,
    *,
    base: float = math.e,
) -> torch.Tensor:
    """Return the maximum ordered singleton conditional temporal MVGC."""
    return pairwise_temporal_mvgc(system, base=base).amax(dim=(-2, -1))


def _partial_covariance(
    covariance: torch.Tensor,
    variables,
    conditioned,
) -> torch.Tensor:
    """Return ``Cov(variables | conditioned)`` by an SPD Schur complement."""
    variable_index = torch.as_tensor(tuple(variables), dtype=torch.long, device=covariance.device)
    conditioned_index = torch.as_tensor(tuple(conditioned), dtype=torch.long, device=covariance.device)
    vv = covariance.index_select(-2, variable_index).index_select(-1, variable_index)
    if conditioned_index.numel() == 0:
        return _hermitian(vv)
    vc = covariance.index_select(-2, variable_index).index_select(-1, conditioned_index)
    cc = covariance.index_select(-2, conditioned_index).index_select(-1, conditioned_index)
    identity = torch.eye(cc.shape[-1], dtype=cc.dtype, device=cc.device).expand_as(cc)
    conditional = vv - vc @ spd_solve(cc, identity) @ vc.transpose(-1, -2)
    return _hermitian(conditional)


def _innovations_inverse_transfer_function(
    system: InnovationsStateSpace,
    frequencies: torch.Tensor,
) -> torch.Tensor:
    """Evaluate ``H(z)^{-1}=I-C(zI-(A-KC))^{-1}K``."""
    transition = system.transition
    observation = system.observation
    gain = system.gain
    single = transition.ndim == 2
    if single:
        transition = transition.unsqueeze(0)
        observation = observation.unsqueeze(0)
        gain = gain.unsqueeze(0)
    batch = max(x.shape[0] for x in (transition, observation, gain))
    transition, observation, gain = [
        x.expand(batch, *x.shape[1:]) if x.shape[0] == 1 else x
        for x in (transition, observation, gain)
    ]
    frequencies = torch.as_tensor(
        frequencies, dtype=transition.dtype, device=transition.device
    )
    complex_dtype = torch.complex128 if transition.dtype == torch.float64 else torch.complex64
    a = transition.to(complex_dtype)
    c = observation.to(complex_dtype)
    k = gain.to(complex_dtype)
    closed_loop = a - k @ c
    state_identity = torch.eye(a.shape[-1], dtype=complex_dtype, device=a.device)
    observation_identity = torch.eye(c.shape[-2], dtype=complex_dtype, device=a.device)
    z = torch.exp(2j * torch.pi * frequencies).reshape(1, -1, 1, 1)
    resolvent = torch.linalg.solve(
        z * state_identity - closed_loop[:, None], k[:, None]
    )
    inverse = observation_identity - c[:, None] @ resolvent
    return inverse[0] if single else inverse


def state_space_spectral_mvgc(
    system: VARSystem | InnovationsStateSpace,
    source,
    target,
    frequencies: torch.Tensor,
    *,
    conditional=None,
    base: float = math.e,
) -> torch.Tensor:
    """Exact conditional spectral GC using the Geweke/MVGC construction.

    By default, ``conditional=None`` follows the ComplexBox/MVGC convention and
    conditions on all variables not listed in ``target`` or ``source``. Pass
    ``conditional=()`` explicitly to marginalise those variables and compute an
    unconditioned source-to-target spectrum.

    For the partition ``X=target``, ``Y=source`` and ``Z=conditional``, the
    full process is first marginalised to ``(X,Z,Y)``. A reduced innovations
    model for ``(X,Z)`` is then obtained by the generalized DARE. At each
    frequency,

    .. math::

       f_{Y\to X\mid Z}(\omega)
       = \log\frac{\det \Sigma^R_{XX}}
       {\det(\Sigma^R_{XX}-Q(\omega)Q(\omega)^*)},

    where ``Q = B^R_{X,:} H_{(X,Z),(Y,Z)} P``; ``B^R`` is the inverse
    transfer function of the reduced model and ``P P^T`` is the partial
    covariance of the full innovations ``(Y,Z)`` conditioned on ``X``.
    """
    innovations = _as_innovations(system)
    target, source, conditional = _normalise_partition(innovations, target, source, conditional)

    full = reduce_innovations_state_space(
        innovations, target + conditional + source
    )
    n_target = len(target)
    n_conditional = len(conditional)
    n_source = len(source)
    target_pos = tuple(range(n_target))
    conditional_pos = tuple(range(n_target, n_target + n_conditional))
    source_pos = tuple(range(n_target + n_conditional, n_target + n_conditional + n_source))
    reduced_pos = target_pos + conditional_pos
    driver_pos = source_pos + conditional_pos

    reduced = reduce_innovations_state_space(full, reduced_pos)
    full_transfer = innovations_transfer_function(full, frequencies)
    reduced_inverse = _innovations_inverse_transfer_function(reduced, frequencies)

    full_covariance = full.innovation_covariance
    reduced_covariance = reduced.innovation_covariance
    single = full_transfer.ndim == 3
    if single:
        full_transfer = full_transfer.unsqueeze(0)
        reduced_inverse = reduced_inverse.unsqueeze(0)
        full_covariance = full_covariance.unsqueeze(0)
        reduced_covariance = reduced_covariance.unsqueeze(0)

    reduced_target_covariance = reduced_covariance[..., :n_target, :n_target]
    partial = _partial_covariance(full_covariance, driver_pos, target_pos)
    partial_cholesky = torch.linalg.cholesky(partial)

    reduced_rows = torch.as_tensor(reduced_pos, dtype=torch.long, device=full_transfer.device)
    driver_columns = torch.as_tensor(driver_pos, dtype=torch.long, device=full_transfer.device)
    full_rw = full_transfer.index_select(-2, reduced_rows).index_select(-1, driver_columns)
    inverse_x = reduced_inverse[..., :n_target, :]
    q = inverse_x @ full_rw @ partial_cholesky[:, None].to(full_transfer.dtype)
    correction = q @ q.conj().transpose(-1, -2)
    denominator = _hermitian(
        reduced_target_covariance[:, None].to(full_transfer.dtype) - correction
    )
    numerator = reduced_target_covariance[:, None].expand(
        -1, denominator.shape[1], -1, -1
    )
    value = (
        torch.linalg.slogdet(numerator).logabsdet
        - torch.linalg.slogdet(denominator).logabsdet
    ) / math.log(base)
    return value[0] if single else value


def integrate_spectral_mvgc(values: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
    """Integrate one-sided spectral GC to the time-domain value.

    For normalized :math:`f\in[0,1/2]`,

    .. math::

       F=2\int_0^{1/2} f_{Y\to X}(\nu)\,d\nu.

    References
    ----------
    - Geweke (1982); Barnett and Seth (2014).
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
