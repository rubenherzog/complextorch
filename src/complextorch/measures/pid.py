"""Gaussian two-source Partial Information Decomposition (PID).

The decomposition is defined for two (possibly multivariate) Gaussian sources
and one (possibly multivariate) Gaussian target.  It reuses the Gaussian
redundancy primitives already required by PhiID so PID and PhiID cannot drift
onto separate mathematical implementations.

References
----------
- Williams, P. L. and Beer, R. D. (2010).
- Barrett, A. B. (2015), Phys. Rev. E 91, 052802.
- Ince, R. A. A. (2017), Entropy 19, 318.
- James, R. G. et al. (2018), J. Phys. A 52, 014002.
- Liardi, A. et al. (2025), PLoS Comput. Biol. 21, e1013629.
"""
from __future__ import annotations

from typing import Literal

import torch

from ..representations import VARSystem
from ._gaussian_redundancy import _mi, _single_target_redundancy, _sobol_gaussian_samples

PIDRedundancy = Literal["mmi", "ccs", "dep"]


def _validate_pid_covariance(
    joint_covariance: torch.Tensor,
    n_source0: int,
    n_source1: int,
) -> torch.Tensor:
    """Validate a batched ``[source0, source1, target]`` covariance."""
    covariance = torch.as_tensor(joint_covariance)
    if not covariance.is_floating_point():
        covariance = covariance.to(torch.get_default_dtype())
    if covariance.ndim < 2 or covariance.shape[-2] != covariance.shape[-1]:
        raise ValueError("joint_covariance must be square on its final two dimensions")
    if n_source0 < 1 or n_source1 < 1:
        raise ValueError("source dimensions must be positive")
    n_target = covariance.shape[-1] - n_source0 - n_source1
    if n_target < 1:
        raise ValueError("joint_covariance must contain a non-empty target block")
    if not bool(torch.isfinite(covariance).all()):
        raise ValueError("joint_covariance must contain only finite values")
    torch.linalg.cholesky(covariance)
    return covariance


def gaussian_pid(
    joint_covariance: torch.Tensor,
    n_source0: int,
    n_source1: int,
    *,
    redundancy: PIDRedundancy | str = "mmi",
    base: float = 2.0,
    ccs_qmc_samples: int = 4096,
) -> dict[str, torch.Tensor]:
    r"""Return the four Gaussian PID atoms for two sources and one target.

    ``joint_covariance`` is ordered ``[source0, source1, target]``.  If
    :math:`R` denotes redundancy, the remaining atoms are obtained from the
    defining PID identities

    .. math::
       U_0 = I(S_0;T)-R,\quad
       U_1 = I(S_1;T)-R,\quad
       S = I(S_0,S_1;T)-U_0-U_1-R.

    Leading batch dimensions are preserved.
    """
    covariance = _validate_pid_covariance(joint_covariance, n_source0, n_source1)
    if base <= 0 or base == 1:
        raise ValueError("base must be positive and different from one")
    method = str(redundancy).lower()
    if method not in {"mmi", "ccs", "dep"}:
        raise ValueError("redundancy must be one of 'mmi', 'ccs', or 'dep'")
    if ccs_qmc_samples < 32:
        raise ValueError("ccs_qmc_samples must be at least 32")

    n_total = covariance.shape[-1]
    source0 = tuple(range(n_source0))
    source1 = tuple(range(n_source0, n_source0 + n_source1))
    target = tuple(range(n_source0 + n_source1, n_total))
    samples = (
        _sobol_gaussian_samples(covariance, ccs_qmc_samples)
        if method == "ccs"
        else None
    )
    phid_method = "idep_a" if method == "dep" else method
    redundant = _single_target_redundancy(
        covariance,
        source0,
        source1,
        target,
        redundancy=phid_method,  # type: ignore[arg-type]
        base=base,
        ccs_samples=samples,
    )
    mi0 = _mi(covariance, source0, target, base=base)
    mi1 = _mi(covariance, source1, target, base=base)
    total = _mi(covariance, source0 + source1, target, base=base)
    unique0 = mi0 - redundant
    unique1 = mi1 - redundant
    synergistic = total - unique0 - unique1 - redundant
    reconstruction = redundant + unique0 + unique1 + synergistic
    return {
        "redundant": redundant,
        "unique_source0": unique0,
        "unique_source1": unique1,
        "synergistic": synergistic,
        "total": total,
        "reconstruction": reconstruction,
    }


def var_past_future_covariance(
    model: VARSystem,
    source0: tuple[int, ...],
    source1: tuple[int, ...],
) -> tuple[torch.Tensor, int, int]:
    """Build ``[source histories, joint future]`` covariance for a VAR(p).

    Each source contains the complete ``p``-lag history of its selected observed
    channels.  The target is the joint next state of all observed channels, as
    in the VAR NuMIT construction of Liardi et al. (2025).
    """
    if not source0 or not source1:
        raise ValueError("source groups must be non-empty")
    if set(source0) & set(source1):
        raise ValueError("source groups must be disjoint")
    n = model.n_variables
    if min(source0 + source1) < 0 or max(source0 + source1) >= n:
        raise ValueError("source index out of range")
    p = model.order
    history0 = tuple(lag * n + channel for lag in range(p) for channel in source0)
    history1 = tuple(lag * n + channel for lag in range(p) for channel in source1)
    indices = torch.as_tensor(
        history0 + history1, dtype=torch.long, device=model.coefficients.device
    )
    history_covariance = model.state_covariance.index_select(-2, indices).index_select(-1, indices)
    history_future = (
        model.state_covariance
        @ model.companion.transpose(-1, -2)
        @ model.projection.transpose(-1, -2)
    ).index_select(-2, indices)
    top = torch.cat([history_covariance, history_future], dim=-1)
    bottom = torch.cat([history_future.transpose(-1, -2), model.present_covariance], dim=-1)
    covariance = torch.cat([top, bottom], dim=-2)
    return covariance, len(history0), len(history1)


def gaussian_pid_from_var(
    model: VARSystem,
    source0: tuple[int, ...],
    source1: tuple[int, ...],
    *,
    redundancy: PIDRedundancy | str = "mmi",
    base: float = 2.0,
    ccs_qmc_samples: int = 4096,
) -> dict[str, torch.Tensor]:
    """Compute Gaussian PID from a canonical VAR model without observations."""
    covariance, n0, n1 = var_past_future_covariance(model, source0, source1)
    return gaussian_pid(
        covariance,
        n0,
        n1,
        redundancy=redundancy,
        base=base,
        ccs_qmc_samples=ccs_qmc_samples,
    )


__all__ = [
    "PIDRedundancy",
    "gaussian_pid",
    "gaussian_pid_from_var",
    "var_past_future_covariance",
]
