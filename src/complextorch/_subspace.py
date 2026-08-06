r"""Internal Bauer SVC and Larimore CVA primitives for model-order selection.

The Larimore route estimates canonical correlations between block-Hankel past
and future vectors. Bauer's singular-value criterion (SVC) then selects the
latent state dimension from the unnormalised canonical correlations.

For candidate dimension :math:`r`, ComplexTorch follows the criterion used by
ComplexBox/MVGC-compatible workflows,

.. math::

   \operatorname{SVC}(r)
   = \rho_{r+1}^{2}
     + \frac{2 n_y r\log N_{\mathrm{eff}}}{N_{\mathrm{eff}}},

where :math:`\rho_{r+1}` is the first canonical correlation omitted by a
rank-:math:`r` model, :math:`n_y` is the number of observed variables and
:math:`N_{\mathrm{eff}}` is the number of past/future Hankel columns.

References
----------
- Bauer, D. (2001). Order estimation for subspace methods. *Automatica*,
  37(10), 1561--1573. https://doi.org/10.1016/S0005-1098(01)00118-2
- Larimore, W. E. (1990). Canonical variate analysis in identification,
  filtering, and adaptive control. *Proceedings of the 29th IEEE Conference
  on Decision and Control*, 596--604. https://doi.org/10.1109/CDC.1990.203665
- Larimore, W. E. (1996). Statistical optimality and canonical variate
  analysis system identification. *Signal Processing*, 52(2), 131--144.
  https://doi.org/10.1016/0165-1684(96)00049-7
- ComplexBox reference implementation:
  https://github.com/bmilinkovic/complexbox
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from ._typing import ArrayLike


def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    """Resolve a public dtype argument to a floating-point Torch dtype."""

    if isinstance(dtype, torch.dtype):
        resolved = dtype
    else:
        try:
            resolved = getattr(torch, dtype)
        except AttributeError as error:
            raise ValueError(f"unknown Torch dtype {dtype!r}") from error
    if resolved not in {torch.float32, torch.float64}:
        raise ValueError("dtype must be torch.float32 or torch.float64")
    return resolved


def _normalise_observations(
    observations: ArrayLike,
    *,
    device: str | torch.device,
    dtype: str | torch.dtype,
) -> tuple[torch.Tensor, bool]:
    """Normalize observations to ``(batch, time, variables)``."""

    source = torch.as_tensor(observations)
    target_device = source.device if device == "auto" else torch.device(device)
    values = source.to(device=target_device, dtype=_resolve_dtype(dtype))
    unbatched = values.ndim == 2
    if unbatched:
        values = values.unsqueeze(0)
    if values.ndim != 3:
        raise ValueError(
            "observations must have shape (time, variables) or "
            "(batch, time, variables)"
        )
    if not torch.isfinite(values).all():
        raise ValueError("observations must be finite")
    return values, unbatched


def _bauer_svc(
    canonical_correlations: ArrayLike,
    n_observations: int,
    n_effective: int | torch.Tensor,
    *,
    min_order: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Select state dimension with Bauer's singular-value criterion.

    Parameters
    ----------
    canonical_correlations
        Unnormalised canonical correlations in descending order, with shape
        ``(..., r_max)``. Leading dimensions are processed independently.
    n_observations
        Number :math:`n_y` of observed variables.
    n_effective
        Effective number :math:`N_{\mathrm{eff}}` of Hankel columns. A scalar
        or tensor broadcastable over the leading batch dimensions.
    min_order
        Smallest candidate state dimension. Defaults to ``n_observations``;
        values below ``n_observations`` are rejected to preserve the
        MVGC/ComplexBox full-model convention.

    Returns
    -------
    best_order
        Tensor containing the minimizing state dimension for each batch.
    criterion
        SVC values with shape ``(..., r_max - min_order + 1)``.

    Notes
    -----
    The correlations are clipped to ``[0, 1]`` only to remove floating-point
    excursions. They must not be divided by the leading correlation before
    applying SVC.

    References
    ----------
    - Bauer (2001), Equation-based singular-value order criteria.
    - ComplexBox ``mvgc.modelorder.bauer_svc``.
    """

    rho = torch.as_tensor(canonical_correlations)
    if rho.ndim < 1 or rho.shape[-1] == 0:
        raise ValueError("canonical_correlations must end in a non-empty axis")
    if not rho.is_floating_point():
        rho = rho.to(torch.float64)
    if not torch.isfinite(rho).all():
        raise ValueError("canonical_correlations must be finite")
    if n_observations < 1:
        raise ValueError("n_observations must be positive")

    r_max = rho.shape[-1]
    lower = n_observations if min_order is None else int(min_order)
    if lower < n_observations:
        raise ValueError("min_order must be at least n_observations")
    if lower > r_max:
        raise ValueError(
            f"minimum order {lower} exceeds maximum identifiable order {r_max}"
        )

    effective = torch.as_tensor(
        n_effective, dtype=rho.dtype, device=rho.device
    )
    if torch.any(effective <= 1) or not torch.isfinite(effective).all():
        raise ValueError("n_effective must be finite and greater than one")

    orders = torch.arange(lower, r_max + 1, device=rho.device)
    # Append the theoretical zero omitted correlation for the full-rank model;
    # indexing at candidate r then retrieves rho_{r+1} in one-based notation.
    padded = torch.cat((rho.clamp(0.0, 1.0), torch.zeros_like(rho[..., :1])), -1)
    omitted = padded.index_select(-1, orders)

    while effective.ndim < rho.ndim - 1:
        effective = effective.unsqueeze(-1)
    penalty = (
        2.0
        * float(n_observations)
        * orders.to(rho.dtype)
        * torch.log(effective.unsqueeze(-1))
        / effective.unsqueeze(-1)
    )
    criterion = omitted.square() + penalty
    best_order = orders[criterion.argmin(dim=-1)]
    return best_order, criterion


@dataclass(frozen=True)
class _StateSpaceOrderComputation:
    """Result of Larimore CVA followed by Bauer SVC.

    Attributes
    ----------
    best_order
        Selected state dimension. Scalar for pooled mode and one value per
        trajectory for independent mode.
    candidate_orders
        Candidate dimensions corresponding to the final criterion axis.
    criterion
        Bauer SVC values.
    canonical_correlations
        Unnormalised canonical correlations from the whitened past/future
        cross-covariance.
    normalized_canonical_correlations
        Correlations divided by the leading value for plotting only.
    n_effective
        Number of Hankel columns entering each estimate.
    """

    best_order: torch.Tensor
    candidate_orders: torch.Tensor
    criterion: torch.Tensor
    canonical_correlations: torch.Tensor
    normalized_canonical_correlations: torch.Tensor
    n_effective: torch.Tensor


def _block_hankel(
    observations: torch.Tensor,
    past_horizon: int,
    future_horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct recent-to-distant past and forward future Hankel blocks."""

    window = observations.unfold(
        dimension=1,
        size=past_horizon + future_horizon,
        step=1,
    )
    # ``unfold`` produces (batch, columns, variables, window). Reverse only the
    # past block so its first row is y_{t-1}, matching Larimore/MVGC notation.
    past = (
        window[..., :past_horizon]
        .flip(-1)
        .permute(0, 3, 2, 1)
        .reshape(observations.shape[0], -1, window.shape[1])
    )
    future = (
        window[..., past_horizon:]
        .permute(0, 3, 2, 1)
        .reshape(observations.shape[0], -1, window.shape[1])
    )
    return past, future


def _canonical_correlations(
    past: torch.Tensor,
    future: torch.Tensor,
    *,
    ridge: float,
) -> torch.Tensor:
    r"""Compute CCA singular values using Cholesky whitening.

    The whitened cross-covariance is

    .. math::

       M=L_f^{-1}\Sigma_{fp}L_p^{-\top},

    where :math:`L_pL_p^\top=\Sigma_{pp}` and
    :math:`L_fL_f^\top=\Sigma_{ff}`. Its singular values are the canonical
    correlations.
    """

    n_effective = past.shape[-1]
    covariance_past = past @ past.transpose(-1, -2) / n_effective
    covariance_future = future @ future.transpose(-1, -2) / n_effective
    cross_covariance = past @ future.transpose(-1, -2) / n_effective

    identity_past = torch.eye(
        covariance_past.shape[-1],
        dtype=past.dtype,
        device=past.device,
    )
    identity_future = torch.eye(
        covariance_future.shape[-1],
        dtype=future.dtype,
        device=future.device,
    )
    # Ridge regularization is restricted to the whitening covariances; it does
    # not alter the cross-covariance or the subsequent Bauer criterion.
    cholesky_past = torch.linalg.cholesky(
        covariance_past + ridge * identity_past
    )
    cholesky_future = torch.linalg.cholesky(
        covariance_future + ridge * identity_future
    )
    left_whitened = torch.linalg.solve_triangular(
        cholesky_future,
        cross_covariance.transpose(-1, -2),
        upper=False,
    )
    whitened = torch.linalg.solve_triangular(
        cholesky_past,
        left_whitened.transpose(-1, -2),
        upper=False,
    ).transpose(-1, -2)
    return torch.linalg.svdvals(whitened)


def _larimore_state_space_order(
    observations: ArrayLike,
    past_horizon: int,
    *,
    future_horizon: int | None = None,
    min_order: int | None = None,
    mode: Literal["pooled", "independent"] = "pooled",
    ridge: float = 1e-12,
    device: str | torch.device = "auto",
    dtype: str | torch.dtype = "float64",
) -> _StateSpaceOrderComputation:
    r"""Estimate full state-space order by Larimore CVA and Bauer SVC.

    Parameters
    ----------
    observations
        Time series in ``(time, variables)`` or ComplexTorch batch-first
        ``(batch, time, variables)`` layout.
    past_horizon, future_horizon
        Numbers of block rows in the past and future Hankel matrices. The
        future horizon defaults to the past horizon.
    min_order
        Smallest candidate state dimension. Defaults to the number of observed
        variables, as required by the full-model MVGC workflow.
    mode
        ``"pooled"`` concatenates Hankel columns across trajectories, matching
        ComplexBox trials. ``"independent"`` computes one order per batch.
    ridge
        Non-negative diagonal regularizer used only for Cholesky whitening.
    device, dtype
        Torch execution device and floating-point dtype.

    Returns
    -------
    _StateSpaceOrderComputation
        Canonical correlations, SVC curve and selected state order.

    References
    ----------
    - Larimore (1990, 1996), canonical variate state construction.
    - Bauer (2001), singular-value model-order estimation.
    - ComplexBox ``mvgc.modelorder.tsdata_to_ssmo``.
    """

    if past_horizon < 1:
        raise ValueError("past_horizon must be positive")
    future = past_horizon if future_horizon is None else int(future_horizon)
    if future < 1:
        raise ValueError("future_horizon must be positive")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    if mode not in {"pooled", "independent"}:
        raise ValueError("mode must be 'pooled' or 'independent'")

    values, _ = _normalise_observations(
        observations, device=device, dtype=dtype
    )
    n_batch, n_times, n_variables = values.shape
    if past_horizon + future > n_times:
        raise ValueError("past/future horizons are too large for the series")

    # Demean each trajectory independently before pooling, matching the
    # independent-trial convention used in MVGC and ComplexBox.
    values = values - values.mean(dim=1, keepdim=True)
    past, future_blocks = _block_hankel(values, past_horizon, future)
    columns_per_trajectory = past.shape[-1]

    if mode == "pooled":
        past = past.permute(1, 0, 2).reshape(past.shape[1], -1).unsqueeze(0)
        future_blocks = (
            future_blocks.permute(1, 0, 2)
            .reshape(future_blocks.shape[1], -1)
            .unsqueeze(0)
        )
        n_effective = torch.tensor(
            n_batch * columns_per_trajectory,
            device=values.device,
            dtype=values.dtype,
        )
    else:
        n_effective = torch.full(
            (n_batch,),
            columns_per_trajectory,
            device=values.device,
            dtype=values.dtype,
        )

    correlations = _canonical_correlations(
        past, future_blocks, ridge=ridge
    )
    r_max = correlations.shape[-1]
    lower = n_variables if min_order is None else int(min_order)
    best_order, criterion = _bauer_svc(
        correlations,
        n_observations=n_variables,
        n_effective=n_effective,
        min_order=lower,
    )
    candidate_orders = torch.arange(
        lower, r_max + 1, device=values.device
    )

    leading = correlations[..., :1]
    normalized = torch.where(
        leading > 0,
        correlations / leading,
        correlations,
    )
    if mode == "pooled":
        best_order = best_order.squeeze(0)
        criterion = criterion.squeeze(0)
        correlations = correlations.squeeze(0)
        normalized = normalized.squeeze(0)

    return _StateSpaceOrderComputation(
        best_order=best_order,
        candidate_orders=candidate_orders,
        criterion=criterion,
        canonical_correlations=correlations,
        normalized_canonical_correlations=normalized,
        n_effective=n_effective,
    )


def _larimore_decomposition(past, future, *, ridge):
    """Return Larimore canonical correlations, right vectors and past factor."""
    n_effective = past.shape[-1]
    covariance_past = past @ past.transpose(-1, -2) / n_effective
    covariance_future = future @ future.transpose(-1, -2) / n_effective
    cross_covariance = past @ future.transpose(-1, -2) / n_effective
    ip = torch.eye(covariance_past.shape[-1], dtype=past.dtype, device=past.device)
    iff = torch.eye(covariance_future.shape[-1], dtype=future.dtype, device=future.device)
    lp = torch.linalg.cholesky(covariance_past + ridge * ip)
    lf = torch.linalg.cholesky(covariance_future + ridge * iff)
    left = torch.linalg.solve_triangular(lf, cross_covariance.transpose(-1, -2), upper=False)
    whitened = torch.linalg.solve_triangular(
        lp, left.transpose(-1, -2), upper=False
    ).transpose(-1, -2)
    _, correlations, right = torch.linalg.svd(whitened, full_matrices=False)
    return correlations, right, lp

