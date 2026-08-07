"""Gaussian temporal and spectral information-rate primitives.

These functions implement the bivariate block quantities used by Faes OIR/HOP
without introducing any O-information or PID public API. Temporal rates are
computed directly from reduced innovations covariances; spectral rates are
computed independently from spectral-density/transfer-function identities.
"""
from __future__ import annotations

import math

import torch

from ..control import (
    InnovationsStateSpace,
    innovations_transfer_function,
    reduce_innovations_state_space,
)
from ..linalg import spd_logdet
from ..spectra import hermitian_logdet, hermitian_part, innovations_spectral_density
from ._nested_var import normalise_indices


def _validate_log_base(base: float) -> float:
    """Return a finite positive logarithm base different from one."""
    value = float(base)
    if not math.isfinite(value) or value <= 0.0 or value == 1.0:
        raise ValueError("base must be finite, positive, and different from one")
    return value


def _normalise_group(indices, n_observations: int, *, name: str) -> tuple[int, ...]:
    """Return a validated non-empty group while rejecting duplicate indices."""
    raw = (indices,) if isinstance(indices, int) else tuple(indices)
    try:
        group = normalise_indices(raw, n_observations)
    except IndexError as exc:
        raise ValueError(f"{name} contains an out-of-range index") from exc
    if len(group) != len(raw):
        raise ValueError(f"{name} must not contain duplicate indices")
    return group


def _normalise_pair(
    system: InnovationsStateSpace,
    left,
    right,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Validate two non-overlapping observation groups."""
    n_observations = system.observation.shape[-2]
    left_group = _normalise_group(left, n_observations, name="left")
    right_group = _normalise_group(right, n_observations, name="right")
    if set(left_group) & set(right_group):
        raise ValueError("left and right groups must be disjoint")
    return left_group, right_group


def gaussian_mutual_information_rate(
    system: InnovationsStateSpace,
    left,
    right,
    *,
    base: float = math.e,
) -> torch.Tensor:
    r"""Return Gaussian mutual-information rate from innovations covariances.

    For stationary Gaussian blocks ``X`` and ``Y``, entropy-rate constants
    cancel and

    .. math::

       \dot I(X;Y)=\frac12\log
       \frac{|V_X|\,|V_Y|}{|V_{XY}|},

    where each ``V`` is the innovations covariance of the corresponding exact
    marginal process.

    Parameters
    ----------
    system
        Exact innovations-form process.
    left, right
        Disjoint non-empty observation-index groups.
    base
        Logarithm base. Defaults to natural units.

    Returns
    -------
    torch.Tensor
        Scalar for an unbatched system or one value per batch element.
    """
    base_value = _validate_log_base(base)
    left_group, right_group = _normalise_pair(system, left, right)
    left_model = reduce_innovations_state_space(system, left_group)
    right_model = reduce_innovations_state_space(system, right_group)
    joint_model = reduce_innovations_state_space(system, left_group + right_group)
    return 0.5 * (
        spd_logdet(left_model.innovation_covariance)
        + spd_logdet(right_model.innovation_covariance)
        - spd_logdet(joint_model.innovation_covariance)
    ) / math.log(base_value)


def gaussian_transfer_entropy_rate(
    system: InnovationsStateSpace,
    source,
    target,
    *,
    base: float = math.e,
) -> torch.Tensor:
    r"""Return Gaussian transfer-entropy rate ``source -> target`` directly.

    The target-only innovations covariance ``V_T^R`` is compared with the
    target block of the joint source-target innovations covariance ``V_T``:

    .. math::

       \dot T_{S\to T}=\frac12\log\frac{|V_T^R|}{|V_T|}.

    This is one half of time-domain Gaussian Granger causality.

    Parameters
    ----------
    system
        Exact innovations-form process.
    source, target
        Disjoint non-empty observation-index groups.
    base
        Logarithm base. Defaults to natural units.

    Returns
    -------
    torch.Tensor
        Scalar for an unbatched system or one value per batch element.
    """
    base_value = _validate_log_base(base)
    source_group, target_group = _normalise_pair(system, source, target)
    joint = reduce_innovations_state_space(system, target_group + source_group)
    target_only = reduce_innovations_state_space(system, target_group)
    n_target = len(target_group)
    joint_target_covariance = joint.innovation_covariance[..., :n_target, :n_target]
    return 0.5 * (
        spd_logdet(target_only.innovation_covariance)
        - spd_logdet(joint_target_covariance)
    ) / math.log(base_value)


def gaussian_instantaneous_information_rate(
    system: InnovationsStateSpace,
    left,
    right,
    *,
    base: float = math.e,
) -> torch.Tensor:
    r"""Return the instantaneous component of Gaussian block MIR.

    For the exact joint innovations covariance ``V`` partitioned into ``X,Y``,

    .. math::

       \dot I_{X\circ Y}=\frac12\log
       \frac{|V_{XX}|\,|V_{YY}|}{|V|}.

    Parameters
    ----------
    system
        Exact innovations-form process.
    left, right
        Disjoint non-empty observation-index groups.
    base
        Logarithm base. Defaults to natural units.

    Returns
    -------
    torch.Tensor
        Scalar for an unbatched system or one value per batch element.
    """
    base_value = _validate_log_base(base)
    left_group, right_group = _normalise_pair(system, left, right)
    joint = reduce_innovations_state_space(system, left_group + right_group)
    n_left = len(left_group)
    covariance = joint.innovation_covariance
    left_covariance = covariance[..., :n_left, :n_left]
    right_covariance = covariance[..., n_left:, n_left:]
    return 0.5 * (
        spd_logdet(left_covariance)
        + spd_logdet(right_covariance)
        - spd_logdet(covariance)
    ) / math.log(base_value)


def spectral_gaussian_mutual_information_rate(
    system: InnovationsStateSpace,
    left,
    right,
    frequencies: torch.Tensor,
    *,
    sampling_frequency: float = 1.0,
    base: float = math.e,
) -> torch.Tensor:
    r"""Return the Gaussian spectral mutual-information-rate density.

    .. math::

       i_{X;Y}(f)=\frac12\log
       \frac{|S_X(f)|\,|S_Y(f)|}{|S_{XY}(f)|}.

    This is the ``f12`` quantity in Faes ``hop_fdTE.m`` after its ``0.5``
    conversion from the OIR coupling convention.

    Parameters
    ----------
    system
        Exact innovations-form process.
    left, right
        Disjoint non-empty observation-index groups.
    frequencies
        One-dimensional frequency grid in cycles per unit time.
    sampling_frequency
        Sampling frequency associated with ``frequencies``.
    base
        Logarithm base. Defaults to natural units.

    Returns
    -------
    torch.Tensor
        Spectral rate density on the final frequency axis.
    """
    base_value = _validate_log_base(base)
    left_group, right_group = _normalise_pair(system, left, right)
    joint = reduce_innovations_state_space(system, left_group + right_group)
    spectrum = innovations_spectral_density(
        joint, frequencies, sampling_frequency=sampling_frequency
    )
    n_left = len(left_group)
    left_spectrum = spectrum[..., :n_left, :n_left]
    right_spectrum = spectrum[..., n_left:, n_left:]
    return 0.5 * (
        hermitian_logdet(left_spectrum)
        + hermitian_logdet(right_spectrum)
        - hermitian_logdet(spectrum)
    ) / math.log(base_value)


def spectral_gaussian_transfer_entropy_rate(
    system: InnovationsStateSpace,
    source,
    target,
    frequencies: torch.Tensor,
    *,
    sampling_frequency: float = 1.0,
    base: float = math.e,
) -> torch.Tensor:
    r"""Return Faes/HOP bivariate spectral TE density ``source -> target``.

    After reducing to the ordered process ``(target, source)``, Faes
    ``hop_fdTE.m`` evaluates

    .. math::

       t_{S\to T}(f)=\frac12\log\frac{|S_T(f)|}
       {|H_{TT}(f)V_{TT}H_{TT}(f)^*|}.

    The frequency integral is independently testable against
    :func:`gaussian_transfer_entropy_rate`.

    Parameters
    ----------
    system
        Exact innovations-form process.
    source, target
        Disjoint non-empty observation-index groups.
    frequencies
        One-dimensional frequency grid in cycles per unit time.
    sampling_frequency
        Sampling frequency associated with ``frequencies``.
    base
        Logarithm base. Defaults to natural units.

    Returns
    -------
    torch.Tensor
        Spectral transfer-entropy density on the final frequency axis.
    """
    base_value = _validate_log_base(base)
    if not math.isfinite(float(sampling_frequency)) or sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be finite and positive")
    source_group, target_group = _normalise_pair(system, source, target)
    joint = reduce_innovations_state_space(system, target_group + source_group)
    frequency = torch.as_tensor(
        frequencies, dtype=joint.transition.dtype, device=joint.transition.device
    )
    transfer = innovations_transfer_function(
        joint, frequency / float(sampling_frequency)
    )
    spectrum = innovations_spectral_density(
        joint, frequency, sampling_frequency=sampling_frequency
    )
    covariance = joint.innovation_covariance
    n_target = len(target_group)
    target_spectrum = spectrum[..., :n_target, :n_target]
    target_covariance = covariance[..., :n_target, :n_target]
    target_transfer = transfer[..., :n_target, :n_target]

    if transfer.ndim == 3:
        intrinsic = (
            target_transfer
            @ target_covariance.to(transfer.dtype)
            @ target_transfer.conj().transpose(-1, -2)
        ) / float(sampling_frequency)
    else:
        if target_covariance.ndim == 2:
            target_covariance = target_covariance.unsqueeze(0)
        intrinsic = (
            target_transfer
            @ target_covariance[:, None].to(transfer.dtype)
            @ target_transfer.conj().transpose(-1, -2)
        ) / float(sampling_frequency)
    intrinsic = hermitian_part(intrinsic)
    return 0.5 * (
        hermitian_logdet(target_spectrum) - hermitian_logdet(intrinsic)
    ) / math.log(base_value)
