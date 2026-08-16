"""Covariance-memory measures for stationary Gaussian dynamics.

CMem1 contrasts full-system predictive information with the active information
storage (AIS) of each component's own history. CMem3 instead contrasts the
full-system past dependence with dependence of each present component on the
full multivariate past. These quantities coincide only in special cases.

References
----------
- Cover, T. M. and Thomas, J. A. (2006). Gaussian information identities.
- ComplexTorch methodological tests and model-backbone implementation.
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
    """Container for CMem totals, curves, lag decomposition, and TC terms."""

    cmem3_total: torch.Tensor
    cmem1_total: torch.Tensor
    cmem3_lag: torch.Tensor
    cmem3_curve: torch.Tensor
    cmem1_curve: torch.Tensor
    tc_innovation: torch.Tensor
    tc_present: torch.Tensor


def _batched_gamma(values: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Normalize autocovariances to ``(batch, lag, n, n)``."""
    gamma = torch.as_tensor(values)
    single = gamma.ndim == 3
    if single:
        gamma = gamma.unsqueeze(0)
    if gamma.ndim != 4 or gamma.shape[-1] != gamma.shape[-2]:
        raise ValueError(
            "autocovariances must have shape (lag,n,n) or (batch,lag,n,n)"
        )
    return gamma, single


def _batched_covariance(values: torch.Tensor, batch: int, n: int) -> torch.Tensor:
    """Normalize a covariance matrix to the requested batch shape."""
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
    r"""Return CMem3 total, :math:`TC(\Sigma_\varepsilon)-TC(\Sigma_X)`."""
    return total_correlation(innovation_covariance) - total_correlation(
        observation_covariance
    )


def cmem1_total_from_covariances(
    observation_covariance: torch.Tensor,
    innovation_covariance: torch.Tensor,
) -> torch.Tensor:
    """Reject covariance-only CMem1 evaluation.

    CMem1 requires each component's own lagged history. Present and innovation
    covariances alone only determine the full-system predictive information;
    using them for the component terms incorrectly collapses CMem1 onto CMem3.
    """
    raise ValueError(
        "CMem1 cannot be computed from present and innovation covariances alone; "
        "self-history autocovariances are required"
    )


def _self_history_ais(
    autocovariances: torch.Tensor,
    history_lag: int,
    *,
    base: float = 2.0,
) -> torch.Tensor:
    r"""Return per-component :math:`I(X_t^i;X_{t-1:t-p}^i)`."""
    gamma, single = _batched_gamma(autocovariances)
    if history_lag < 1:
        raise ValueError("history_lag must be at least one")
    if gamma.shape[1] <= history_lag:
        raise ValueError("autocovariances do not contain the requested history")
    batch, _, n, _ = gamma.shape
    values = []
    for node in range(n):
        history = torch.empty(
            (batch, history_lag, history_lag),
            dtype=gamma.dtype,
            device=gamma.device,
        )
        for left in range(history_lag):
            for right in range(history_lag):
                history[:, left, right] = gamma[:, abs(right - left), node, node]
        cross = torch.stack(
            [gamma[:, lag, node, node] for lag in range(1, history_lag + 1)],
            -1,
        )
        current = gamma[:, 0, node, node].reshape(batch, 1, 1)
        joint = torch.cat(
            [
                torch.cat([current, cross.unsqueeze(-2)], -1),
                torch.cat([cross.unsqueeze(-1), history], -1),
            ],
            -2,
        )
        values.append(gaussian_mutual_information(joint, 1, base=base))
    result = torch.stack(values, -1)
    return result[0] if single else result


def cmem1_total_from_primitives(
    observation_covariance: torch.Tensor,
    innovation_covariance: torch.Tensor,
    autocovariances: torch.Tensor,
    history_lag: int,
) -> torch.Tensor:
    r"""Return CMem1 using full-system PI and component self-history AIS.

    .. math::

       CMem_1 = I(X_t;X_{t-1:t-p})
                - \sum_i I(X_t^i;X_{t-1:t-p}^i).
    """
    gamma, single = _batched_gamma(autocovariances)
    batch, _, n, _ = gamma.shape
    present = _batched_covariance(observation_covariance, batch, n)
    innovations = _batched_covariance(innovation_covariance, batch, n)
    full = 0.5 * (spd_logdet(present) - spd_logdet(innovations)) / math.log(2.0)
    parts = _self_history_ais(gamma, history_lag)
    if parts.ndim == 1:
        parts = parts.unsqueeze(0)
    result = full - parts.sum(-1)
    return result[0] if single else result


def _joint_from_gamma(sigma: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """Build covariance of ``(X_t, X_{t-tau})``."""
    return torch.cat(
        [
            torch.cat([sigma, gamma.transpose(-1, -2)], -1),
            torch.cat([gamma, sigma], -1),
        ],
        -2,
    )


def _self_mi(joint: torch.Tensor, n: int) -> torch.Tensor:
    """Return singleton self mutual informations from a two-time joint."""
    a = torch.diagonal(joint[..., :n, :n], dim1=-2, dim2=-1)
    b = torch.diagonal(joint[..., n:, n:], dim1=-2, dim2=-1)
    c = torch.diagonal(joint[..., :n, n:], dim1=-2, dim2=-1)
    determinant = (a * b - c.square()).clamp_min(torch.finfo(joint.dtype).tiny)
    return 0.5 * torch.log2(a * b / determinant)


def cmem3_curve_from_autocovariances(
    autocovariances: torch.Tensor,
    tau_max: int,
) -> torch.Tensor:
    r"""Return marginal-lag CMem3 curve through ``tau_max``.

    At lag :math:`\tau`, CMem3 is
    :math:`TC(\Sigma_{X_t\mid X_{t-\tau}})-TC(\Sigma_X)`.
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
    r"""Return :math:`I(X_t;X_{t-\tau})-\sum_i I(X_t^i;X_{t-\tau}^i)`."""
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
    """Build covariance of ``(X_t, X_{t-1}, ..., X_{t-tau})``."""
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
    """Return ``I(X_t^i; X_past)`` for each present component."""
    past = joint[..., n:, n:]
    past_logdet = spd_logdet(past)
    values = []
    for node in range(n):
        index = torch.tensor([node, *range(n, 2 * n)], device=joint.device)
        sub = joint.index_select(-2, index).index_select(-1, index)
        values.append(
            0.5
            * (torch.log(joint[..., node, node]) + past_logdet - spd_logdet(sub))
            / math.log(2.0)
        )
    return torch.stack(values, -1)


def _cmem3_lag_one(gammas: torch.Tensor, tau: int) -> torch.Tensor:
    r"""Return the conditional CMem3 contribution of one VAR lag."""
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
    """Return conditional CMem3 contributions for lags ``1..max_lag``."""
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
    """Compute CMem totals, curves, and finite-lag decomposition."""
    gamma, single = _batched_gamma(autocovariances)
    batch, _, n, _ = gamma.shape
    present = _batched_covariance(observation_covariance, batch, n)
    innovations = _batched_covariance(innovation_covariance, batch, n)
    result = CMemResult(
        cmem3_total_from_covariances(present, innovations),
        cmem1_total_from_primitives(
            present, innovations, gamma, decomposition_max_lag
        ),
        cmem3_lag_decomposition_from_autocovariances(gamma, decomposition_max_lag),
        cmem3_curve_from_autocovariances(gamma, curve_max_lag),
        cmem1_curve_from_autocovariances(gamma, curve_max_lag),
        total_correlation(innovations),
        total_correlation(present),
    )
    if not single:
        return result
    return CMemResult(*(value[0] for value in result.__dict__.values()))


def cmem1_full_past(
    model,
    *,
    base: float = 2.0,
    marginal_method: str = "dare",
    frequencies: torch.Tensor | None = None,
    sampling_frequency: float = 1.0,
    half_open: bool = False,
) -> torch.Tensor:
    r"""Return full-past CMem1 from exact or spectral marginal entropy rates.

    The full-past definition is

    .. math::

       CMem_1 = [H(X_t)-h(X)] - \sum_i [H(X_t^i)-h(X^i)],

    where every marginal entropy rate :math:`h(X^i)` is the entropy rate of
    the exact marginal process. ``marginal_method="dare"`` uses exact
    generalized-DARE reductions. ``marginal_method="spectral"`` reuses one
    full spectral density and numerical quadrature, which is substantially
    cheaper for high-dimensional feature extraction.
    """
    from .backbone import as_innovations, observation_autocovariances
    from .entropy_rate import marginal_entropy_rate
    from .gaussian import gaussian_entropy

    innovations = as_innovations(model)
    present = observation_autocovariances(model, 0)[..., 0, :, :]
    full_ais = gaussian_entropy(present, base=base) - gaussian_entropy(
        innovations.innovation_covariance, base=base
    )
    diagonal = torch.diagonal(present, dim1=-2, dim2=-1)
    marginal_present = 0.5 * (
        math.log(2.0 * math.pi * math.e) + torch.log(diagonal)
    ) / math.log(float(base))
    marginal_rates = marginal_entropy_rate(
        innovations,
        base=base,
        method=marginal_method,
        frequencies=frequencies,
        sampling_frequency=sampling_frequency,
        half_open=half_open,
    )
    return full_ais - (marginal_present - marginal_rates).sum(dim=-1)


def cmem3_total(system: VARSystem) -> torch.Tensor:
    """Return CMem3 total for a canonical VAR system."""
    return cmem3_total_from_covariances(
        system.present_covariance, system.innovation_covariance
    )


def cmem1_total(system: VARSystem) -> torch.Tensor:
    """Return CMem1 total using each component's complete VAR self-history."""
    from .backbone import observation_autocovariances

    gamma = observation_autocovariances(system, system.order)
    return cmem1_total_from_primitives(
        system.present_covariance,
        system.innovation_covariance,
        gamma,
        system.order,
    )


def cmem3_curve(
    system: VARSystem,
    tau_max: int,
    *,
    autocovariance_sequence: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the marginal-lag CMem3 curve for a VAR system."""
    from .backbone import observation_autocovariances

    gamma = (
        observation_autocovariances(system, tau_max)
        if autocovariance_sequence is None
        else autocovariance_sequence
    )
    return cmem3_curve_from_autocovariances(gamma, tau_max)


def cmem1_curve(
    system: VARSystem,
    tau_max: int,
    *,
    autocovariance_sequence: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the marginal-lag CMem1 curve for a VAR system."""
    from .backbone import observation_autocovariances

    gamma = (
        observation_autocovariances(system, tau_max)
        if autocovariance_sequence is None
        else autocovariance_sequence
    )
    return cmem1_curve_from_autocovariances(gamma, tau_max)


def cmem3_lag_decomposition(
    system: VARSystem,
    *,
    autocovariance_sequence: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the CMem3 chain-rule decomposition across VAR lags."""
    from .backbone import observation_autocovariances

    gamma = (
        observation_autocovariances(system, system.order)
        if autocovariance_sequence is None
        else autocovariance_sequence
    )
    return cmem3_lag_decomposition_from_autocovariances(gamma, system.order)


def compute_cmem(
    system: VARSystem,
    tau_max: int = 1,
    *,
    autocovariance_sequence: torch.Tensor | None = None,
) -> CMemResult:
    """Compute all CMem outputs for a canonical VAR system."""
    from .backbone import observation_autocovariances

    required = max(tau_max, system.order)
    gamma = (
        observation_autocovariances(system, required)
        if autocovariance_sequence is None
        else autocovariance_sequence
    )
    return compute_cmem_from_primitives(
        system.present_covariance,
        system.innovation_covariance,
        gamma,
        curve_max_lag=tau_max,
        decomposition_max_lag=system.order,
    )


__all__ = [
    "CMemResult",
    "cmem3_total_from_covariances",
    "cmem1_total_from_covariances",
    "cmem1_total_from_primitives",
    "cmem3_curve_from_autocovariances",
    "cmem1_curve_from_autocovariances",
    "cmem3_lag_decomposition_from_autocovariances",
    "compute_cmem_from_primitives",
    "cmem3_total",
    "cmem1_total",
    "cmem1_full_past",
    "cmem3_curve",
    "cmem1_curve",
    "cmem3_lag_decomposition",
    "compute_cmem",
]
