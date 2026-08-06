"""Analytical CMem quantities from model-derived Gaussian covariances.

Notes
-----
CMem quantities decompose covariance-memory effects using Gaussian total
correlation and lagged covariance blocks. All determinants are evaluated with
positive-definite linear algebra.

References
----------
- Cover, T. M. and Thomas, J. A. (2006), Gaussian information identities.
- ComplexTorch repository methodological notes and tests.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from ..linalg import spd_logdet, spd_solve, symmetrise
from ..representations import VARSystem
from .gaussian import gaussian_mutual_information, total_correlation


@dataclass(frozen=True)
class CMemResult:
    """CMemResult.
    
    Notes
    -----
    The class follows the scikit-learn fitted-attribute convention when applicable.
    """
    cmem3_total: torch.Tensor
    cmem1_total: torch.Tensor
    cmem3_lag: torch.Tensor
    cmem3_curve: torch.Tensor
    cmem1_curve: torch.Tensor
    tc_innovation: torch.Tensor
    tc_present: torch.Tensor


def _batched_gamma(values: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """ batched gamma.
    
    Parameters
    ----------
    values
        Input controlling ``_batched_gamma``.
    
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
    gamma = torch.as_tensor(values)
    single = gamma.ndim == 3
    if single:
        gamma = gamma.unsqueeze(0)
    if gamma.ndim != 4 or gamma.shape[-1] != gamma.shape[-2]:
        raise ValueError("autocovariances must have shape (lag,n,n) or (batch,lag,n,n)")
    return gamma, single


def _batched_covariance(values: torch.Tensor, batch: int, n: int) -> torch.Tensor:
    """ batched covariance.
    
    Parameters
    ----------
    values
        Input controlling ``_batched_covariance``.
    batch
        Input controlling ``_batched_covariance``.
    n
        Input controlling ``_batched_covariance``.
    
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
    covariance = torch.as_tensor(values)
    if covariance.ndim == 2:
        covariance = covariance.unsqueeze(0)
    if covariance.ndim != 3 or covariance.shape[-2:] != (n, n):
        raise ValueError("covariance has incompatible shape")
    if covariance.shape[0] == 1 and batch > 1:
        covariance = covariance.expand(batch, -1, -1)
    if covariance.shape[0] != batch:
        raise ValueError("covariance batch dimension is incompatible")
    return covariance


def cmem3_total_from_covariances(
    observation_covariance: torch.Tensor,
    innovation_covariance: torch.Tensor,
) -> torch.Tensor:
    """Cmem3 total from covariances.
    
    Parameters
    ----------
    observation_covariance
        Input controlling ``cmem3_total_from_covariances``.
    innovation_covariance
        Input controlling ``cmem3_total_from_covariances``.
    
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
    return total_correlation(innovation_covariance) - total_correlation(observation_covariance)


def cmem1_total_from_covariances(
    observation_covariance: torch.Tensor,
    innovation_covariance: torch.Tensor,
) -> torch.Tensor:
    """Cmem1 total from covariances.
    
    Parameters
    ----------
    observation_covariance
        Input controlling ``cmem1_total_from_covariances``.
    innovation_covariance
        Input controlling ``cmem1_total_from_covariances``.
    
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
    full = 0.5 * (
        # Evaluate log-determinants through an SPD-aware factorisation for numerical stability.
        spd_logdet(observation_covariance) - spd_logdet(innovation_covariance)
    ) / math.log(2.0)
    parts = 0.5 * torch.log2(
        torch.diagonal(observation_covariance, dim1=-2, dim2=-1)
        / torch.diagonal(innovation_covariance, dim1=-2, dim2=-1)
    ).sum(-1)
    return full - parts


def _joint_from_gamma(sigma: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """ joint from gamma.
    
    Parameters
    ----------
    sigma
        Input controlling ``_joint_from_gamma``.
    gamma
        Input controlling ``_joint_from_gamma``.
    
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
    return torch.cat(
        [
            torch.cat([sigma, gamma.transpose(-1, -2)], -1),
            torch.cat([gamma, sigma], -1),
        ],
        -2,
    )


def _self_mi(joint: torch.Tensor, n: int) -> torch.Tensor:
    """ self mi.
    
    Parameters
    ----------
    joint
        Input controlling ``_self_mi``.
    n
        Input controlling ``_self_mi``.
    
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
    a = torch.diagonal(joint[..., :n, :n], dim1=-2, dim2=-1)
    b = torch.diagonal(joint[..., n:, n:], dim1=-2, dim2=-1)
    c = torch.diagonal(joint[..., :n, n:], dim1=-2, dim2=-1)
    determinant = (a * b - c.square()).clamp_min(torch.finfo(joint.dtype).tiny)
    return 0.5 * torch.log2(a * b / determinant)


def cmem3_curve_from_autocovariances(
    autocovariances: torch.Tensor,
    tau_max: int,
) -> torch.Tensor:
    """Cmem3 curve from autocovariances.
    
    Parameters
    ----------
    autocovariances
        Input controlling ``cmem3_curve_from_autocovariances``.
    tau_max
        Input controlling ``cmem3_curve_from_autocovariances``.
    
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
    if tau_max < 1:
        raise ValueError("tau_max must be at least one")
    gamma, single = _batched_gamma(autocovariances)
    if gamma.shape[1] <= tau_max:
        raise ValueError("autocovariances do not contain the requested lags")
    sigma = gamma[:, 0]
    present_tc = total_correlation(sigma)
    values = []
    for tau in range(1, tau_max + 1):
        conditional = symmetrise(
            sigma
            - gamma[:, tau]
            @ spd_solve(sigma, gamma[:, tau].transpose(-1, -2))
        )
        values.append(total_correlation(conditional) - present_tc)
    result = torch.stack(values, -1)
    return result[0] if single else result


def cmem1_curve_from_autocovariances(
    autocovariances: torch.Tensor,
    tau_max: int,
) -> torch.Tensor:
    """Cmem1 curve from autocovariances.
    
    Parameters
    ----------
    autocovariances
        Input controlling ``cmem1_curve_from_autocovariances``.
    tau_max
        Input controlling ``cmem1_curve_from_autocovariances``.
    
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
    if tau_max < 1:
        raise ValueError("tau_max must be at least one")
    gamma, single = _batched_gamma(autocovariances)
    if gamma.shape[1] <= tau_max:
        raise ValueError("autocovariances do not contain the requested lags")
    n = gamma.shape[-1]
    sigma = gamma[:, 0]
    values = []
    for tau in range(1, tau_max + 1):
        joint = _joint_from_gamma(sigma, gamma[:, tau])
        values.append(
            gaussian_mutual_information(joint, n) - _self_mi(joint, n).sum(-1)
        )
    result = torch.stack(values, -1)
    return result[0] if single else result


def _joint_cov_lags(gammas: torch.Tensor, tau: int) -> torch.Tensor:
    """ joint cov lags.
    
    Parameters
    ----------
    gammas
        Input controlling ``_joint_cov_lags``.
    tau
        Input controlling ``_joint_cov_lags``.
    
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
    blocks = []
    for left in range(tau + 1):
        row = []
        for right in range(tau + 1):
            delta = right - left
            block = gammas[:, abs(delta)]
            row.append(block if delta >= 0 else block.transpose(-1, -2))
        blocks.append(torch.cat(row, -1))
    return torch.cat(blocks, -2)


def _node_vs_vector_mi(joint: torch.Tensor, n: int) -> torch.Tensor:
    """ node vs vector mi.
    
    Parameters
    ----------
    joint
        Input controlling ``_node_vs_vector_mi``.
    n
        Input controlling ``_node_vs_vector_mi``.
    
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
    past = joint[..., n:, n:]
    # Evaluate log-determinants through an SPD-aware factorisation for numerical stability.
    past_logdet = spd_logdet(past)
    values = []
    for node in range(n):
        index = torch.tensor([node, *range(n, 2 * n)], device=joint.device)
        sub = joint.index_select(-2, index).index_select(-1, index)
        values.append(
            0.5
            # Evaluate log-determinants through an SPD-aware factorisation for numerical stability.
            * (torch.log(joint[..., node, node]) + past_logdet - spd_logdet(sub))
            / math.log(2.0)
        )
    return torch.stack(values, -1)


def _cmem3_lag_one(gammas: torch.Tensor, tau: int) -> torch.Tensor:
    """ cmem3 lag one.
    
    Parameters
    ----------
    gammas
        Input controlling ``_cmem3_lag_one``.
    tau
        Input controlling ``_cmem3_lag_one``.
    
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
    n = gammas.shape[-1]
    full = _joint_cov_lags(gammas, tau)
    present = torch.arange(0, n, device=full.device)
    target = torch.arange(tau * n, (tau + 1) * n, device=full.device)
    joint_index = torch.cat([present, target])
    if tau == 1:
        conditional = full.index_select(-2, joint_index).index_select(-1, joint_index)
    else:
        condition = torch.arange(n, tau * n, device=full.device)
        ordered = torch.cat([joint_index, condition])
        reordered = full.index_select(-2, ordered).index_select(-1, ordered)
        n_joint = 2 * n
        s11 = reordered[..., :n_joint, :n_joint]
        s12 = reordered[..., :n_joint, n_joint:]
        s22 = reordered[..., n_joint:, n_joint:]
        conditional = symmetrise(
            s11 - s12 @ spd_solve(s22, s12.transpose(-1, -2))
        )
    return (
        gaussian_mutual_information(conditional, n)
        - _node_vs_vector_mi(conditional, n).sum(-1)
    )


def cmem3_lag_decomposition_from_autocovariances(
    autocovariances: torch.Tensor,
    max_lag: int,
) -> torch.Tensor:
    """Cmem3 lag decomposition from autocovariances.
    
    Parameters
    ----------
    autocovariances
        Input controlling ``cmem3_lag_decomposition_from_autocovariances``.
    max_lag
        Input controlling ``cmem3_lag_decomposition_from_autocovariances``.
    
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
    if max_lag < 1:
        raise ValueError("max_lag must be at least one")
    gammas, single = _batched_gamma(autocovariances)
    if gammas.shape[1] <= max_lag:
        raise ValueError("autocovariances do not contain the requested lags")
    result = torch.stack(
        [_cmem3_lag_one(gammas, tau) for tau in range(1, max_lag + 1)], -1
    )
    return result[0] if single else result


def compute_cmem_from_primitives(
    observation_covariance: torch.Tensor,
    innovation_covariance: torch.Tensor,
    autocovariances: torch.Tensor,
    *,
    curve_max_lag: int = 1,
    decomposition_max_lag: int = 1,
) -> CMemResult:
    """Compute cmem from primitives.
    
    Parameters
    ----------
    observation_covariance
        Input controlling ``compute_cmem_from_primitives``.
    innovation_covariance
        Input controlling ``compute_cmem_from_primitives``.
    autocovariances
        Input controlling ``compute_cmem_from_primitives``.
    curve_max_lag
        Input controlling ``compute_cmem_from_primitives``.
    decomposition_max_lag
        Input controlling ``compute_cmem_from_primitives``.
    
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
    gamma, single = _batched_gamma(autocovariances)
    batch, _, n, _ = gamma.shape
    present = _batched_covariance(observation_covariance, batch, n)
    innovations = _batched_covariance(innovation_covariance, batch, n)
    result = CMemResult(
        cmem3_total_from_covariances(present, innovations),
        cmem1_total_from_covariances(present, innovations),
        cmem3_lag_decomposition_from_autocovariances(gamma, decomposition_max_lag),
        cmem3_curve_from_autocovariances(gamma, curve_max_lag),
        cmem1_curve_from_autocovariances(gamma, curve_max_lag),
        total_correlation(innovations),
        total_correlation(present),
    )
    if not single:
        return result
    return CMemResult(*(value[0] for value in result.__dict__.values()))


# Backward-compatible VAR wrappers.
def cmem3_total(system: VARSystem) -> torch.Tensor:
    """Cmem3 total.
    
    Parameters
    ----------
    system
        Input controlling ``cmem3_total``.
    
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
    return cmem3_total_from_covariances(
        system.present_covariance, system.innovation_covariance
    )


def cmem1_total(system: VARSystem) -> torch.Tensor:
    """Cmem1 total.
    
    Parameters
    ----------
    system
        Input controlling ``cmem1_total``.
    
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
    return cmem1_total_from_covariances(
        system.present_covariance, system.innovation_covariance
    )


def cmem3_curve(
    system: VARSystem,
    tau_max: int,
    *,
    autocovariance_sequence: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cmem3 curve.
    
    Parameters
    ----------
    system
        Input controlling ``cmem3_curve``.
    tau_max
        Input controlling ``cmem3_curve``.
    autocovariance_sequence
        Input controlling ``cmem3_curve``.
    
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
    from .backbone import observation_autocovariances
    gamma = (
        observation_autocovariances(system, tau_max)
        if autocovariance_sequence is None else autocovariance_sequence
    )
    return cmem3_curve_from_autocovariances(gamma, tau_max)


def cmem1_curve(
    system: VARSystem,
    tau_max: int,
    *,
    autocovariance_sequence: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cmem1 curve.
    
    Parameters
    ----------
    system
        Input controlling ``cmem1_curve``.
    tau_max
        Input controlling ``cmem1_curve``.
    autocovariance_sequence
        Input controlling ``cmem1_curve``.
    
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
    from .backbone import observation_autocovariances
    gamma = (
        observation_autocovariances(system, tau_max)
        if autocovariance_sequence is None else autocovariance_sequence
    )
    return cmem1_curve_from_autocovariances(gamma, tau_max)


def cmem3_lag_decomposition(
    system: VARSystem,
    *,
    autocovariance_sequence: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cmem3 lag decomposition.
    
    Parameters
    ----------
    system
        Input controlling ``cmem3_lag_decomposition``.
    autocovariance_sequence
        Input controlling ``cmem3_lag_decomposition``.
    
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
    from .backbone import observation_autocovariances
    gamma = (
        observation_autocovariances(system, system.order)
        if autocovariance_sequence is None else autocovariance_sequence
    )
    return cmem3_lag_decomposition_from_autocovariances(gamma, system.order)


def compute_cmem(
    system: VARSystem,
    tau_max: int = 1,
    *,
    autocovariance_sequence: torch.Tensor | None = None,
) -> CMemResult:
    """Compute cmem.
    
    Parameters
    ----------
    system
        Input controlling ``compute_cmem``.
    tau_max
        Input controlling ``compute_cmem``.
    autocovariance_sequence
        Input controlling ``compute_cmem``.
    
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
    from .backbone import observation_autocovariances
    required = max(tau_max, system.order)
    gamma = (
        observation_autocovariances(system, required)
        if autocovariance_sequence is None else autocovariance_sequence
    )
    return compute_cmem_from_primitives(
        system.present_covariance,
        system.innovation_covariance,
        gamma,
        curve_max_lag=tau_max,
        decomposition_max_lag=system.order,
    )


__all__ = [
    "CMemResult", "cmem3_total_from_covariances", "cmem1_total_from_covariances",
    "cmem3_curve_from_autocovariances", "cmem1_curve_from_autocovariances",
    "cmem3_lag_decomposition_from_autocovariances", "compute_cmem_from_primitives",
    "cmem3_total", "cmem1_total", "cmem3_curve", "cmem1_curve",
    "cmem3_lag_decomposition", "compute_cmem",
]
