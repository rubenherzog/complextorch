"""Analytical CMem quantities for stationary Gaussian VAR systems."""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from ..linalg import spd_logdet, spd_solve, symmetrise
from ..representations import VARSystem
from .gaussian import gaussian_mutual_information, total_correlation


@dataclass(frozen=True)
class CMemResult:
    cmem3_total: torch.Tensor
    cmem1_total: torch.Tensor
    cmem3_lag: torch.Tensor
    cmem3_curve: torch.Tensor
    cmem1_curve: torch.Tensor
    tc_innovation: torch.Tensor
    tc_present: torch.Tensor


def cmem3_total(system: VARSystem) -> torch.Tensor:
    return total_correlation(system.innovation_covariance) - total_correlation(system.present_covariance)


def cmem1_total(system: VARSystem) -> torch.Tensor:
    sigma = system.present_covariance
    q = system.innovation_covariance
    full = 0.5 * (spd_logdet(sigma) - spd_logdet(q)) / math.log(2)
    parts = 0.5 * torch.log2(
        torch.diagonal(sigma, dim1=-2, dim2=-1)
        / torch.diagonal(q, dim1=-2, dim2=-1)
    ).sum(-1)
    return full - parts


def _validate_autocovariances(system: VARSystem, values: torch.Tensor, max_lag: int) -> torch.Tensor:
    gamma = torch.as_tensor(values, dtype=system.coefficients.dtype, device=system.coefficients.device)
    expected = (system.batch_size, max_lag + 1, system.n_variables, system.n_variables)
    if gamma.ndim != 4 or gamma.shape[0] != expected[0] or gamma.shape[2:] != expected[2:] or gamma.shape[1] < expected[1]:
        raise ValueError("autocovariance_sequence has incompatible shape or insufficient lags")
    return gamma


def _compute_autocovariances(system: VARSystem, max_lag: int) -> torch.Tensor:
    power = torch.eye(
        system.companion.shape[-1],
        dtype=system.companion.dtype,
        device=system.companion.device,
    ).expand(system.batch_size, -1, -1)
    values = []
    for lag in range(max_lag + 1):
        if lag:
            power = power @ system.companion
        values.append(
            system.projection
            @ power
            @ system.state_covariance
            @ system.projection.transpose(-1, -2)
        )
    return torch.stack(values, dim=1)


def _autocovariances(system: VARSystem, max_lag: int, supplied: torch.Tensor | None) -> torch.Tensor:
    if supplied is None:
        return _compute_autocovariances(system, max_lag)
    return _validate_autocovariances(system, supplied, max_lag)


def _joint_from_gamma(sigma: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            torch.cat([sigma, gamma], -1),
            torch.cat([gamma.transpose(-1, -2), sigma], -1),
        ],
        -2,
    )


def cmem3_curve(
    system: VARSystem,
    tau_max: int,
    *,
    autocovariance_sequence: torch.Tensor | None = None,
) -> torch.Tensor:
    if tau_max < 1:
        raise ValueError("tau_max must be >=1")
    gamma = _autocovariances(system, tau_max, autocovariance_sequence)
    sigma = gamma[:, 0]
    tc = total_correlation(sigma)
    values = []
    for tau in range(1, tau_max + 1):
        conditional = symmetrise(
            sigma
            - gamma[:, tau]
            @ spd_solve(sigma, gamma[:, tau].transpose(-1, -2))
        )
        values.append(total_correlation(conditional) - tc)
    return torch.stack(values, -1)


def _self_mi(joint: torch.Tensor, n: int) -> torch.Tensor:
    a = torch.diagonal(joint[..., :n, :n], dim1=-2, dim2=-1)
    b = torch.diagonal(joint[..., n:, n:], dim1=-2, dim2=-1)
    c = torch.diagonal(joint[..., :n, n:], dim1=-2, dim2=-1)
    det = (a * b - c.square()).clamp_min(torch.finfo(joint.dtype).tiny)
    return 0.5 * torch.log2(a * b / det)


def cmem1_curve(
    system: VARSystem,
    tau_max: int,
    *,
    autocovariance_sequence: torch.Tensor | None = None,
) -> torch.Tensor:
    if tau_max < 1:
        raise ValueError("tau_max must be >=1")
    gamma = _autocovariances(system, tau_max, autocovariance_sequence)
    n = system.n_variables
    sigma = gamma[:, 0]
    values = []
    for tau in range(1, tau_max + 1):
        joint = _joint_from_gamma(sigma, gamma[:, tau])
        values.append(
            gaussian_mutual_information(joint, n) - _self_mi(joint, n).sum(-1)
        )
    return torch.stack(values, -1)


def _joint_cov_lags(gammas: torch.Tensor, tau: int) -> torch.Tensor:
    blocks = []
    for left in range(tau + 1):
        row = []
        for right in range(tau + 1):
            block = gammas[:, abs(right - left)]
            row.append(block.transpose(-1, -2) if right < left else block)
        blocks.append(torch.cat(row, -1))
    return torch.cat(blocks, -2)


def _node_vs_vector_mi(joint: torch.Tensor, n: int) -> torch.Tensor:
    past = joint[..., n:, n:]
    ld = spd_logdet(past)
    values = []
    for node in range(n):
        index = torch.tensor([node, *range(n, 2 * n)], device=joint.device)
        sub = joint.index_select(-2, index).index_select(-1, index)
        values.append(
            0.5
            * (torch.log(joint[..., node, node]) + ld - spd_logdet(sub))
            / math.log(2)
        )
    return torch.stack(values, -1)


def _cmem3_lag_one(system: VARSystem, tau: int, gammas: torch.Tensor) -> torch.Tensor:
    n = system.n_variables
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


def cmem3_lag_decomposition(
    system: VARSystem,
    *,
    autocovariance_sequence: torch.Tensor | None = None,
) -> torch.Tensor:
    gammas = _autocovariances(system, system.order, autocovariance_sequence)
    return torch.stack(
        [_cmem3_lag_one(system, tau, gammas) for tau in range(1, system.order + 1)],
        -1,
    )


def compute_cmem(
    system: VARSystem,
    tau_max: int = 1,
    *,
    autocovariance_sequence: torch.Tensor | None = None,
) -> CMemResult:
    required_lag = max(tau_max, system.order)
    gammas = _autocovariances(system, required_lag, autocovariance_sequence)
    return CMemResult(
        cmem3_total(system),
        cmem1_total(system),
        cmem3_lag_decomposition(system, autocovariance_sequence=gammas),
        cmem3_curve(system, tau_max, autocovariance_sequence=gammas),
        cmem1_curve(system, tau_max, autocovariance_sequence=gammas),
        total_correlation(system.innovation_covariance),
        total_correlation(system.present_covariance),
    )
