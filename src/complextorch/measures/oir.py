r"""Gaussian O-information rate for stationary random processes.

The O-information rate (OIR) extends the O-information of Rosas et al. from
random variables to stationary random processes. For ``N`` process groups,

.. math::

   \dot\Omega(X_1,\ldots,X_N)
   =(N-2)\dot H(X)
   +\sum_{i=1}^N\left[\dot H(X_i)-\dot H(X_{-i})\right].

For Gaussian innovations models, entropy-rate constants cancel exactly, so the
time-domain implementation depends only on log-determinants of the exact
marginal innovations covariances. The frequency-domain implementation is an
independent spectral representation based on exact marginal spectral-density
matrices. Positive OIR is redundancy-dominated and negative OIR is
synergy-dominated.

All leading batch dimensions of :class:`InnovationsStateSpace` are preserved.
Python loops are only over the small combinatorial collection of process groups;
all state-space reductions, DARE solves, log-determinants, and spectral matrix
operations remain Torch-batched over trajectories/systems.

References
----------
- Rosas, F. E. et al. (2019). Quantifying high-order interdependencies via the
  O-information. *Physical Review E*, 100, 032305.
- Faes, L. et al. (2022). A new framework for the time- and frequency-domain
  assessment of high-order interactions in networks of random processes.
  *IEEE Transactions on Signal Processing*, 70, 5766-5777.
- Scagliarini, T. et al. (2023). Gradients of O-information: Low-order
  descriptors of high-order dependencies. *Physical Review Research*, 5,
  013025.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from ..control import InnovationsStateSpace, reduce_innovations_state_space
from ..linalg import spd_logdet
from ..spectra import hermitian_logdet, innovations_spectral_density
from .rates import (
    _normalise_group,
    _validate_log_base,
    gaussian_mutual_information_rate,
    spectral_gaussian_mutual_information_rate,
)

Group = tuple[int, ...]


def _normalise_groups(
    system: InnovationsStateSpace,
    groups: Sequence[int | Sequence[int]] | None,
) -> tuple[Group, ...]:
    """Validate disjoint non-empty process groups without touching batch axes."""
    n_observations = system.observation.shape[-2]
    if groups is None:
        normalised = tuple((index,) for index in range(n_observations))
    else:
        raw_groups = tuple(groups)
        if not raw_groups:
            raise ValueError("groups must contain at least two non-empty groups")
        normalised = tuple(
            _normalise_group(group, n_observations, name=f"groups[{index}]")
            for index, group in enumerate(raw_groups)
        )
    if len(normalised) < 2:
        raise ValueError("O-information rate requires at least two groups")
    flattened = tuple(index for group in normalised for index in group)
    if len(set(flattened)) != len(flattened):
        raise ValueError("groups must be pairwise disjoint")
    return normalised


def _flatten(groups: Sequence[Group]) -> Group:
    """Concatenate observation groups while preserving their declared order."""
    return tuple(index for group in groups for index in group)


def _marginal_model(
    system: InnovationsStateSpace,
    indices: Group,
    cache: dict[Group, InnovationsStateSpace],
) -> InnovationsStateSpace:
    """Return and cache one exact marginal innovations model."""
    model = cache.get(indices)
    if model is None:
        model = reduce_innovations_state_space(system, indices)
        cache[indices] = model
    return model


def _oir_logdet_terms(
    system: InnovationsStateSpace,
    groups: tuple[Group, ...],
) -> torch.Tensor:
    """Evaluate the temporal OIR log-determinant identity in natural units."""
    cache: dict[Group, InnovationsStateSpace] = {}
    all_indices = _flatten(groups)
    joint = spd_logdet(
        _marginal_model(system, all_indices, cache).innovation_covariance
    )
    singleton_terms = torch.stack(
        [
            spd_logdet(_marginal_model(system, group, cache).innovation_covariance)
            for group in groups
        ],
        dim=-1,
    ).sum(dim=-1)
    leave_one_out_terms = torch.stack(
        [
            spd_logdet(
                _marginal_model(
                    system,
                    _flatten(groups[:index] + groups[index + 1 :]),
                    cache,
                ).innovation_covariance
            )
            for index in range(len(groups))
        ],
        dim=-1,
    ).sum(dim=-1)
    return singleton_terms - leave_one_out_terms + (len(groups) - 2) * joint


def o_information_rate(
    system: InnovationsStateSpace,
    groups: Sequence[int | Sequence[int]] | None = None,
    *,
    base: float = math.e,
) -> torch.Tensor:
    r"""Compute the Gaussian O-information rate in the time domain.

    Parameters
    ----------
    system
        Exact innovations-form state-space process. Unbatched and batched
        systems are supported without flattening leading batch dimensions.
    groups
        Disjoint process groups. Each entry is an observation index or a
        sequence of indices. ``None`` treats every observation channel as one
        group. Channels not listed are marginalized out exactly.
    base
        Logarithm base. Defaults to natural units, matching the Faes OIR
        convention.

    Returns
    -------
    torch.Tensor
        Scalar for an unbatched system, otherwise one OIR value per leading
        batch element.

    Notes
    -----
    With two groups the defining identity is exactly zero. For three or more
    groups, positive values indicate redundancy-dominated dependence and
    negative values indicate synergy-dominated dependence.
    """
    base_value = _validate_log_base(base)
    normalised = _normalise_groups(system, groups)
    return 0.5 * _oir_logdet_terms(system, normalised) / math.log(base_value)


def _spectral_oir_logdet_terms(
    system: InnovationsStateSpace,
    groups: tuple[Group, ...],
    frequencies: torch.Tensor,
    *,
    sampling_frequency: float,
    marginalization: str,
) -> torch.Tensor:
    """Evaluate the OIR spectral log-determinant identity at every frequency."""
    cache: dict[Group, InnovationsStateSpace] = {}
    full_spectrum = None
    if marginalization == "spectrum":
        full_spectrum = innovations_spectral_density(
            system, frequencies, sampling_frequency=sampling_frequency
        )
    elif marginalization != "dare":
        raise ValueError("marginalization must be 'dare' or 'spectrum'")

    def log_spectrum(indices: Group) -> torch.Tensor:
        """Return the Hermitian log-determinant spectrum of one marginal."""
        if full_spectrum is not None:
            index = torch.as_tensor(indices, dtype=torch.long, device=full_spectrum.device)
            spectrum = full_spectrum.index_select(-2, index).index_select(-1, index)
        else:
            model = _marginal_model(system, indices, cache)
            spectrum = innovations_spectral_density(
                model, frequencies, sampling_frequency=sampling_frequency
            )
        return hermitian_logdet(spectrum)

    joint = log_spectrum(_flatten(groups))
    singleton_terms = torch.stack(
        [log_spectrum(group) for group in groups], dim=-1
    ).sum(dim=-1)
    leave_one_out_terms = torch.stack(
        [
            log_spectrum(_flatten(groups[:index] + groups[index + 1 :]))
            for index in range(len(groups))
        ],
        dim=-1,
    ).sum(dim=-1)
    return singleton_terms - leave_one_out_terms + (len(groups) - 2) * joint


def spectral_o_information_rate(
    system: InnovationsStateSpace,
    frequencies: torch.Tensor,
    groups: Sequence[int | Sequence[int]] | None = None,
    *,
    sampling_frequency: float = 1.0,
    base: float = math.e,
    marginalization: str = "dare",
) -> torch.Tensor:
    r"""Compute the frequency-resolved Gaussian O-information rate.

    Parameters
    ----------
    system
        Exact innovations-form process, possibly with leading batch dimensions.
    frequencies
        One-dimensional frequencies in physical units, on the same device as
        the system or convertible by the shared spectral primitive.
    groups
        Disjoint process groups. ``None`` uses every observation channel as a
        singleton group.
    sampling_frequency
        Positive sampling frequency passed to the shared innovations-spectrum
        implementation.
    base
        Logarithm base. Defaults to natural units.
    marginalization
        ``"dare"`` preserves the exact reduced-innovations path.
        ``"spectrum"`` obtains every marginal spectral matrix as a submatrix
        of one full model spectrum, avoiding repeated DARE solves while giving
        the same frequency-resolved quantity.

    Returns
    -------
    torch.Tensor
        Frequency-resolved OIR with all system batch dimensions preserved and
        frequency on the final non-matrix axis.

    Notes
    -----
    Whole-band integration with :func:`complextorch.integrate_spectral_rate`
    recovers :func:`o_information_rate` up to numerical quadrature error.
    """
    base_value = _validate_log_base(base)
    normalised = _normalise_groups(system, groups)
    return (
        0.5
        * _spectral_oir_logdet_terms(
            system,
            normalised,
            frequencies,
            sampling_frequency=sampling_frequency,
            marginalization=marginalization,
        )
        / math.log(base_value)
    )


def _normalise_target_group(target_group: int, n_groups: int) -> int:
    """Validate an index selecting one group from an OIR group collection."""
    if not isinstance(target_group, int):
        raise TypeError("target_group must be an integer group position")
    if not 0 <= target_group < n_groups:
        raise ValueError("target_group is out of range")
    return target_group


def delta_o_information_rate(
    system: InnovationsStateSpace,
    target_group: int,
    groups: Sequence[int | Sequence[int]] | None = None,
    *,
    base: float = math.e,
) -> torch.Tensor:
    r"""Return the O-information gradient associated with one process group.

    For ``N`` groups and selected group ``j``, Faes et al.'s rate form is

    .. math::

       \Delta\dot\Omega_j
       =(2-N)\dot I(X_j;X_{-j})
       +\sum_{m\ne j}\dot I(X_j;X_{-\{j,m\}}).

    This is exactly ``OIR(groups) - OIR(groups without j)``.

    Parameters
    ----------
    system
        Exact innovations-form process, batched or unbatched.
    target_group
        Zero-based position in ``groups``. When ``groups=None`` this is also
        the observation-channel index.
    groups
        At least three disjoint process groups.
    base
        Logarithm base. Defaults to natural units.

    Returns
    -------
    torch.Tensor
        Scalar for an unbatched system or one value per leading batch element.
    """
    normalised = _normalise_groups(system, groups)
    if len(normalised) < 3:
        raise ValueError("delta O-information rate requires at least three groups")
    target_position = _normalise_target_group(target_group, len(normalised))
    target = normalised[target_position]
    others = normalised[:target_position] + normalised[target_position + 1 :]
    all_others = _flatten(others)
    value = (2 - len(normalised)) * gaussian_mutual_information_rate(
        system, target, all_others, base=base
    )
    for excluded in range(len(others)):
        remaining = _flatten(others[:excluded] + others[excluded + 1 :])
        value = value + gaussian_mutual_information_rate(
            system, target, remaining, base=base
        )
    return value


def spectral_delta_o_information_rate(
    system: InnovationsStateSpace,
    frequencies: torch.Tensor,
    target_group: int,
    groups: Sequence[int | Sequence[int]] | None = None,
    *,
    sampling_frequency: float = 1.0,
    base: float = math.e,
) -> torch.Tensor:
    r"""Return the frequency-resolved O-information gradient for one group.

    Parameters
    ----------
    system
        Exact innovations-form process, batched or unbatched.
    frequencies
        Frequency grid passed to the shared spectral MIR primitive.
    target_group
        Zero-based position in ``groups``.
    groups
        At least three disjoint process groups. ``None`` uses singleton
        observation channels.
    sampling_frequency
        Positive sampling frequency.
    base
        Logarithm base. Defaults to natural units.

    Returns
    -------
    torch.Tensor
        Frequency-resolved gradient with batch dimensions preserved.

    Notes
    -----
    This function is built from the independently validated spectral MIR
    primitive rather than from :func:`spectral_o_information_rate`, providing
    a separate numerical path for the defining gradient identity.
    """
    normalised = _normalise_groups(system, groups)
    if len(normalised) < 3:
        raise ValueError("spectral delta O-information rate requires at least three groups")
    target_position = _normalise_target_group(target_group, len(normalised))
    target = normalised[target_position]
    others = normalised[:target_position] + normalised[target_position + 1 :]
    all_others = _flatten(others)
    value = (2 - len(normalised)) * spectral_gaussian_mutual_information_rate(
        system,
        target,
        all_others,
        frequencies,
        sampling_frequency=sampling_frequency,
        base=base,
    )
    for excluded in range(len(others)):
        remaining = _flatten(others[:excluded] + others[excluded + 1 :])
        value = value + spectral_gaussian_mutual_information_rate(
            system,
            target,
            remaining,
            frequencies,
            sampling_frequency=sampling_frequency,
            base=base,
        )
    return value
