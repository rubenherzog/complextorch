r"""Shared Larimore subspace-identification primitives.

This private module contains mathematical operations reused by fixed-dimension
state-space fitting and latent-dimension selection. It deliberately owns no
selection policy or public estimator: Bauer SVC and temporal order search live
under :mod:`complextorch.selection`.
"""
from __future__ import annotations

import torch


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


def _block_hankel(
    observations: torch.Tensor,
    past_horizon: int,
    future_horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Construct Larimore past and future block-Hankel matrices.

    For each valid time origin, the past block is ordered recent-to-distant,
    :math:`(y_{t-1}, y_{t-2}, \ldots)`, while the future block is ordered
    forward from :math:`y_t`. Input trajectories remain separate on the leading
    batch axis; no Hankel column can span a trajectory boundary.

    Parameters
    ----------
    observations
        Tensor with shape ``(batch, time, variables)``.
    past_horizon, future_horizon
        Number of observation blocks on each side of the time origin.

    Returns
    -------
    past, future
        Tensors with shapes ``(batch, past_horizon * variables, columns)`` and
        ``(batch, future_horizon * variables, columns)``.
    """
    window = observations.unfold(
        dimension=1,
        size=past_horizon + future_horizon,
        step=1,
    )
    # ``unfold`` gives (batch, columns, variables, window). Reverse only the
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


def _larimore_decomposition(past, future, *, ridge):
    r"""Compute Larimore CVA whitening and its singular-value decomposition.

    Let :math:`P` and :math:`F` be past and future block-Hankel matrices with
    :math:`N` columns. The implementation forms

    .. math::

       \Sigma_{PP}=PP^\top/N,\qquad
       \Sigma_{FF}=FF^\top/N,\qquad
       \Sigma_{FP}=FP^\top/N.

    With Cholesky factors
    :math:`L_PL_P^\top=\Sigma_{PP}+\lambda I` and
    :math:`L_FL_F^\top=\Sigma_{FF}+\lambda I`, the canonical correlations are
    the singular values of

    .. math::

       M=L_F^{-1}\Sigma_{FP}L_P^{-\top}.

    ``ridge`` therefore regularizes only the two whitening covariances. The
    triangular solves below avoid explicit inverses. The returned right
    singular vectors and :math:`L_P` are reused by Larimore state construction.

    References
    ----------
    - Larimore, W. E. (1990). Canonical variate analysis in identification,
      filtering, and adaptive control.
    - Larimore, W. E. (1996). Statistical optimality and canonical variate
      analysis system identification.
    """
    n_effective = past.shape[-1]
    covariance_past = past @ past.transpose(-1, -2) / n_effective
    covariance_future = future @ future.transpose(-1, -2) / n_effective
    # Compute Sigma_PF here; its transpose below is Sigma_FP in the equation.
    cross_covariance = past @ future.transpose(-1, -2) / n_effective

    identity_past = torch.eye(
        covariance_past.shape[-1], dtype=past.dtype, device=past.device
    )
    identity_future = torch.eye(
        covariance_future.shape[-1], dtype=future.dtype, device=future.device
    )
    cholesky_past = torch.linalg.cholesky(
        covariance_past + ridge * identity_past
    )
    cholesky_future = torch.linalg.cholesky(
        covariance_future + ridge * identity_future
    )

    # First solve L_F X = Sigma_FP, then right-whiten by L_P^{-T}; the second
    # triangular solve is transposed to implement that right multiplication.
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

    _, correlations, right_vectors = torch.linalg.svd(
        whitened, full_matrices=False
    )
    return correlations, right_vectors, cholesky_past
