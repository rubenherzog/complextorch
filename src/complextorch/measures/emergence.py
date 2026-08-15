"""Rosas--Mediano practical criteria for causal emergence.

This module implements the finite-delay practical criteria introduced in
Rosas et al. (2020), Eq. (10), for a deterministic linear macro-feature
``V_t = L X_t``.  For stationary Gaussian models the criteria are evaluated
exactly from model-implied autocovariances, without simulation or refitting.

References
----------
- Rosas, F. E., Mediano, P. A. M., Jensen, H. J., Seth, A. K., Barrett, A. B.,
  Carhart-Harris, R. L., and Bor, D. (2020). Reconciling emergences: An
  information-theoretic approach to identify causal emergence in multivariate
  data. *PLoS Computational Biology*, 16(12), e1008289.
- ``pmediano/ReconcilingEmergences`` commit
  ``ecf591aacb6d58996c903b51a2f945cd7f713a32``.
"""
from __future__ import annotations

import math

import torch

from ..linalg import symmetrise
from .backbone import CovarianceModel, observation_autocovariances
from .gaussian import gaussian_mutual_information


def _validate_base(base: float) -> float:
    """Validate and normalize the logarithm base."""
    value = float(base)
    if not math.isfinite(value) or value <= 0.0 or value == 1.0:
        raise ValueError("base must be finite, positive, and different from 1")
    return value


def _batched_autocovariances(
    autocovariances: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    """Normalize autocovariances to ``(batch, lag, n, n)``."""
    gamma = torch.as_tensor(autocovariances)
    single = gamma.ndim == 3
    if single:
        gamma = gamma.unsqueeze(0)
    if gamma.ndim != 4 or gamma.shape[-1] != gamma.shape[-2]:
        raise ValueError(
            "autocovariances must have shape (lag,n,n) or (batch,lag,n,n)"
        )
    return gamma, single


def _batched_projection(
    projection: torch.Tensor,
    covariance: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    """Validate and normalize a linear macro projection to batched form."""
    matrix = torch.as_tensor(
        projection, dtype=covariance.dtype, device=covariance.device
    )
    single = matrix.ndim == 2
    if single:
        matrix = matrix.unsqueeze(0)
    if matrix.ndim != 3:
        raise ValueError("macro_projection must have shape (m,n) or (batch,m,n)")
    if matrix.shape[-1] != covariance.shape[-1]:
        raise ValueError("macro_projection input dimension must match model dimension")
    if not 1 <= matrix.shape[-2] <= matrix.shape[-1]:
        raise ValueError("macro_projection output dimension must be between 1 and n")
    if not bool(torch.isfinite(matrix).all().item()):
        raise ValueError("macro_projection must contain only finite values")
    if bool(torch.any(torch.linalg.matrix_rank(matrix) < matrix.shape[-2]).item()):
        raise ValueError("macro_projection must have full row rank")
    return matrix, single


def _broadcast_model_projection(
    gamma: torch.Tensor,
    projection: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Broadcast model and projection over one compatible batch dimension."""
    batch = max(gamma.shape[0], projection.shape[0])
    if gamma.shape[0] not in (1, batch) or projection.shape[0] not in (1, batch):
        raise ValueError("incompatible model and macro_projection batch dimensions")
    if gamma.shape[0] == 1:
        gamma = gamma.expand(batch, *gamma.shape[1:])
    if projection.shape[0] == 1:
        projection = projection.expand(batch, *projection.shape[1:])
    return gamma, projection


def _scalar_vector_mi(
    scalar_variance: torch.Tensor,
    vector_covariance: torch.Tensor,
    vector_scalar_cross: torch.Tensor,
    *,
    base: float,
) -> torch.Tensor:
    """Vectorized ``I(scalar; vector)`` using the canonical Gaussian-MI core."""
    n = scalar_variance.shape[-1]
    m = vector_covariance.shape[-1]
    vector_covariance = vector_covariance.unsqueeze(-3).expand(-1, n, m, m)
    cross = vector_scalar_cross.transpose(-1, -2).unsqueeze(-2)
    top = torch.cat([scalar_variance[..., None, None], cross], dim=-1)
    bottom = torch.cat([cross.transpose(-1, -2), vector_covariance], dim=-1)
    joint = torch.cat([top, bottom], dim=-2)
    return gaussian_mutual_information(joint, 1, base=base)


def _pairwise_scalar_mi(
    variance: torch.Tensor,
    future_past_cross: torch.Tensor,
    *,
    base: float,
) -> torch.Tensor:
    """Return ``I(X_t^i; X_{t+tau}^j)`` for every ``(j, i)`` pair."""
    n = variance.shape[-1]
    past_variance = variance[:, None, :].expand(-1, n, -1)
    future_variance = variance[:, :, None].expand(-1, -1, n)
    joint = torch.stack(
        [
            torch.stack([past_variance, future_past_cross], dim=-1),
            torch.stack([future_past_cross, future_variance], dim=-1),
        ],
        dim=-2,
    )
    return gaussian_mutual_information(joint, 1, base=base)


def emergence_from_autocovariances(
    autocovariances: torch.Tensor,
    macro_projection: torch.Tensor,
    *,
    lag: int = 1,
    base: float = 2.0,
) -> dict[str, torch.Tensor]:
    r"""Evaluate the Rosas--Mediano practical emergence criteria.

    For ``V_t = L X_t`` and :math:`\tau=\text{lag}` this computes

    .. math::

       \Psi_\tau(V)
       = I(V_t;V_{t+\tau})
       - \sum_j I(X_t^j;V_{t+\tau}),

    .. math::

       \Delta_\tau(V)
       = \max_j\left[
       I(V_t;X_{t+\tau}^j)
       - \sum_i I(X_t^i;X_{t+\tau}^j)\right],

    .. math::

       \Gamma_\tau(V)
       = \max_j I(V_t;X_{t+\tau}^j).

    ``autocovariances[..., tau, :, :]`` follows the ComplexTorch convention
    :math:`\operatorname{Cov}(X_{t+\tau}, X_t)`.
    """
    if lag < 1:
        raise ValueError("lag must be at least one")
    base = _validate_base(base)
    gamma, gamma_single = _batched_autocovariances(autocovariances)
    if gamma.shape[-3] <= lag:
        raise ValueError("autocovariances do not contain the requested lag")
    projection, projection_single = _batched_projection(
        macro_projection, gamma[:, 0]
    )
    gamma, projection = _broadcast_model_projection(gamma, projection)

    present = symmetrise(gamma[:, 0])
    future_past = gamma[:, lag]
    macro_covariance = symmetrise(
        projection @ present @ projection.transpose(-1, -2)
    )
    macro_future_past = projection @ future_past @ projection.transpose(-1, -2)
    macro_joint = torch.cat(
        [
            torch.cat(
                [macro_covariance, macro_future_past.transpose(-1, -2)], dim=-1
            ),
            torch.cat([macro_future_past, macro_covariance], dim=-1),
        ],
        dim=-2,
    )
    macro_mi = gaussian_mutual_information(
        macro_joint, macro_covariance.shape[-1], base=base
    )

    variances = torch.diagonal(present, dim1=-2, dim2=-1)

    # Cov(V_{t+tau}, X_t): shape (batch, macro, micro).
    macro_future_micro_past = projection @ future_past
    micro_to_macro = _scalar_vector_mi(
        variances,
        macro_covariance,
        macro_future_micro_past,
        base=base,
    )

    # Cov(X_{t+tau}, V_t): shape (batch, micro, macro).
    micro_future_macro_past = future_past @ projection.transpose(-1, -2)
    macro_to_micro = _scalar_vector_mi(
        variances,
        macro_covariance,
        micro_future_macro_past.transpose(-1, -2),
        base=base,
    )

    pairwise_micro = _pairwise_scalar_mi(
        variances, future_past, base=base
    )
    micro_sum_by_target = pairwise_micro.sum(dim=-1)

    psi = macro_mi - micro_to_macro.sum(dim=-1)
    delta_terms = macro_to_micro - micro_sum_by_target
    delta = delta_terms.max(dim=-1).values
    gamma_value = macro_to_micro.max(dim=-1).values

    result = {
        "psi": psi,
        "delta": delta,
        "gamma": gamma_value,
        "macro_mutual_information": macro_mi,
        "micro_to_macro_mutual_information": micro_to_macro,
        "macro_to_micro_mutual_information": macro_to_micro,
        "micro_pairwise_mutual_information": pairwise_micro,
    }
    if gamma_single and projection_single:
        return {name: value[0] for name, value in result.items()}
    return result


def emergence_from_model(
    model: CovarianceModel,
    macro_projection: torch.Tensor,
    *,
    lag: int = 1,
    history: str = "lagged",
    base: float = 2.0,
    autocovariance_sequence: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    r"""Return Rosas--Mediano :math:`\Psi`, :math:`\Delta`, and :math:`\Gamma`.

    The published practical criteria are finite-delay quantities. ``history``
    therefore currently accepts only ``"lagged"``. A full-past extension is
    intentionally not exposed yet: exact cross-prediction terms such as
    :math:`I(V_{<t};X_t^j)` require prediction covariances conditioned on an
    arbitrary projected history, not only the reduced innovation covariance.
    They are computable with generalized DAREs, but ComplexTorch does not yet
    expose the reusable projected-history prediction primitive needed to do so
    without duplicating the control core.

    Parameters
    ----------
    model
        Stationary ``VARSystem``, ``StateSpaceModel``, or
        ``InnovationsStateSpace`` supported by the covariance backbone.
    macro_projection
        Full-row-rank linear map ``L`` in ``V_t = L X_t``, with shape
        ``(m,n)`` or ``(batch,m,n)``. No projection is estimated internally.
    lag
        Positive delay :math:`\tau`. Defaults to 1.
    history
        ``"lagged"`` for the published finite-delay criteria. ``"full"`` is
        reserved for a future explicitly ComplexTorch-defined extension and
        currently raises ``NotImplementedError``.
    base
        Logarithm base. Defaults to 2 (bits).
    autocovariance_sequence
        Optional precomputed model autocovariances, used by aggregate measure
        evaluation to avoid recomputation.
    """
    if history not in {"lagged", "full"}:
        raise ValueError("history must be 'lagged' or 'full'")
    if history == "full":
        raise NotImplementedError(
            "history='full' requires projected-history conditional prediction "
            "covariances from generalized DAREs; this extension is not yet "
            "implemented to avoid duplicating the control reduction core"
        )
    autocovariances = (
        observation_autocovariances(model, lag)
        if autocovariance_sequence is None
        else autocovariance_sequence
    )
    return emergence_from_autocovariances(
        autocovariances, macro_projection, lag=lag, base=base
    )


def emergence_measures(
    model: CovarianceModel,
    macro_projection: torch.Tensor,
    *,
    lag: int = 1,
    base: float = 2.0,
) -> dict[str, torch.Tensor]:
    """Backward-compatible alias for :func:`emergence_from_model`."""
    return emergence_from_model(model, macro_projection, lag=lag, base=base)


def emergence_from_observations(
    observations: torch.Tensor,
    macro_projection: torch.Tensor,
    *,
    lag: int = 1,
    base: float = 2.0,
) -> dict[str, torch.Tensor]:
    """Gaussian plug-in estimate of the published finite-delay criteria.

    This secondary estimator exists for compatibility with the observation API;
    the primary implementation is :func:`emergence_from_model`.
    """
    if lag < 1:
        raise ValueError("lag must be at least one")
    x = torch.as_tensor(observations)
    if x.ndim != 2:
        raise ValueError("observations must have shape (time, variables)")
    if x.shape[0] <= lag + 1:
        raise ValueError("observations are too short for the requested lag")
    if not x.is_floating_point():
        raise TypeError("observations must use a floating-point dtype")
    if not bool(torch.isfinite(x).all().item()):
        raise ValueError("observations must contain only finite values")

    centered = x - x.mean(dim=0)
    denominator = x.shape[0] - 1
    gamma0 = centered.transpose(-1, -2) @ centered / denominator
    future = centered[lag:]
    past = centered[:-lag]
    gamma_lag = future.transpose(-1, -2) @ past / (future.shape[0] - 1)
    autocovariances = torch.stack([gamma0, gamma_lag], dim=0)
    if lag > 1:
        padded = torch.zeros(
            (lag + 1, x.shape[-1], x.shape[-1]), dtype=x.dtype, device=x.device
        )
        padded[0] = gamma0
        padded[lag] = gamma_lag
        autocovariances = padded
    return emergence_from_autocovariances(
        autocovariances, macro_projection, lag=lag, base=base
    )


__all__ = [
    "emergence_from_autocovariances",
    "emergence_from_model",
    "emergence_measures",
    "emergence_from_observations",
]
