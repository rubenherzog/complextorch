"""Gaussian temporal and spectral information-rate primitives.

These functions implement the bivariate block quantities used by Faes OIR/HOP
without introducing any O-information or PID public API.  Temporal rates are
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


def _normalise_group(indices, n_observations: int, *, name: str) -> tuple[int, ...]:
    group = tuple(int(index) for index in indices)
    if not group:
        raise ValueError(f"{name} must contain at least one index")
    if len(set(group)) != len(group):
        raise ValueError(f"{name} must not contain duplicate indices")
    if min(group) < 0 or max(group) >= n_observations:
        raise ValueError(f"{name} contains an out-of-range index")
    return group


def _normalise_pair(
    system: InnovationsStateSpace,
    left,
    right,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
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
    """
    if base <= 0 or base == 1:
        raise ValueError("base must be positive and different from one")
    left_group, right_group = _normalise_pair(system, left, right)
    left_model = reduce_innovations_state_space(system, left_group)
    right_model = reduce_innovations_state_space(system, right_group)
    joint_model = reduce_innovations_state_space(system, left_group + right_group)
    return 0.5 * (
        spd_logdet(left_model.innovation_covariance)
        + spd_logdet(right_model.innovation_covariance)
        - spd_logdet(joint_model.innovation_covariance)
    ) / math.log(base)


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
    """
    if base <= 0 or base == 1:
        raise ValueError("base must be positive and different from one")
    source_group, target_group = _normalise_pair(system, source, target)
    joint = reduce_innovations_state_space(system, target_group + source_group)
    target_only = reduce_innovations_state_space(system, target_group)
    n_target = len(target_group)
    joint_target_covariance = joint.innovation_covariance[..., :n_target, :n_target]
    return 0.5 * (
        spd_logdet(target_only.innovation_covariance)
        - spd_logdet(joint_target_covariance)
    ) / math.log(base)


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
    """
    if base <= 0 or base == 1:
        raise ValueError("base must be positive and different from one")
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
    ) / math.log(base)


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
    """
    if base <= 0 or base == 1:
        raise ValueError("base must be positive and different from one")
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
    ) / math.log(base)


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
    """
    if base <= 0 or base == 1:
        raise ValueError("base must be positive and different from one")
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
    ) / math.log(base)
