"""Shared subspace-identification primitives.

This private module contains mathematical operations reused by fixed-order
state-space estimators and state-order selectors. It deliberately does not own
model-selection policy or public estimators.

Larimore CVA whitens past/future block-Hankel covariances and decomposes the
whitened cross-covariance. Selection criteria such as Bauer SVC live under
``complextorch.selection``.
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
