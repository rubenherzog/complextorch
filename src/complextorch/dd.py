"""Dynamical-dependence objectives and projection-level evaluation.

This module defines the DD quantities evaluated for a supplied projection. It
does not search over projections; optimization lives in :mod:`dd_optimization`.
"""
from __future__ import annotations

import torch

from .control import (
    InnovationsStateSpace,
    _as_innovations_state_space,
    dynamical_dependence,
    innovations_transfer_function,
)
from .representations import StateSpaceModel, VARSystem

Model = StateSpaceModel | InnovationsStateSpace | VARSystem


def _single_innovations(system: Model) -> InnovationsStateSpace:
    """Return one innovations system, accepting singleton model batches."""
    iss = _as_innovations_state_space(system)
    parts = []
    for value in (
        iss.transition,
        iss.observation,
        iss.gain,
        iss.innovation_covariance,
    ):
        tensor = torch.as_tensor(value)
        if tensor.ndim == 3:
            if tensor.shape[0] != 1:
                raise ValueError(
                    "DD optimization currently accepts one microscopic system; "
                    "system batching is outside the present parity scope"
                )
            tensor = tensor[0]
        if tensor.ndim != 2:
            raise ValueError("innovations-system matrices must be two-dimensional")
        parts.append(tensor)
    return InnovationsStateSpace(*parts)


def _row_bases(projection: torch.Tensor, n: int) -> tuple[torch.Tensor, bool]:
    """Validate row projections and return shape ``(runs,m,n)``."""
    matrix = torch.as_tensor(projection)
    single = matrix.ndim == 2
    if single:
        matrix = matrix.unsqueeze(0)
    if matrix.ndim != 3:
        raise ValueError("projection must have shape (m,n) or (runs,m,n)")
    if matrix.shape[-1] != n:
        raise ValueError("projection input dimension must match observation dimension")
    if not 1 <= matrix.shape[-2] <= n:
        raise ValueError("projection output dimension must be between 1 and n")
    if matrix.shape[0] < 1:
        raise ValueError("projection must contain at least one run")
    if not bool(torch.isfinite(matrix).all().item()):
        raise ValueError("projection must contain only finite values")
    return matrix, single


def innovations_proxy_sequence(system: Model, lags: int | None = None) -> torch.Tensor:
    r"""Return the SSDI proxy sequence ``Q_k = C A^{k-1} K``.

    Returns
    -------
    torch.Tensor
        Shape ``(lags, n, n)``. If omitted, ``lags`` is the latent state
        dimension, matching ComplexBox/SSDI ``iss2cak``.
    """
    iss = _single_innovations(system)
    a, c, k = iss.transition, iss.observation, iss.gain
    state_dimension = a.shape[-1]
    lags = state_dimension if lags is None else int(lags)
    if lags < 1:
        raise ValueError("lags must be at least 1")
    power = torch.eye(state_dimension, dtype=a.dtype, device=a.device)
    values = []
    for lag in range(lags):
        if lag:
            power = power @ a
        values.append(c @ power @ k)
    return torch.stack(values)


def proxy_dynamical_dependence(
    projection: torch.Tensor,
    sequence: torch.Tensor,
) -> torch.Tensor:
    r"""Evaluate ComplexBox ``cak2ddx`` in row-projection convention.

    For ``M=L.T`` and ``Q_k=C A^{k-1}K``,

    .. math::

       D_x(M)=\sum_k \|M Q_k\|_F^2-\|M Q_k M^T\|_F^2.

    This low-level kernel is the identity-innovation ComplexBox objective.
    Model-level wrappers whiten a general ``V > 0`` before calling it.
    """
    q = torch.as_tensor(sequence)
    if q.ndim != 3 or q.shape[-1] != q.shape[-2]:
        raise ValueError("sequence must have shape (lags,n,n)")
    m, single = _row_bases(projection, q.shape[-1])
    q = q.to(dtype=m.dtype, device=m.device)
    mq = torch.einsum("bmi,kij->bkmj", m, q)
    mqmt = torch.einsum("bkmi,bni->bkmn", mq, m)
    value = torch.sum(mq * mq, dim=(1, 2, 3)) - torch.sum(
        mqmt * mqmt, dim=(1, 2, 3)
    )
    return value[0] if single else value


def proxy_dynamical_dependence_gradient(
    projection: torch.Tensor,
    sequence: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Return the analytic Grassmannian gradient of proxy DD.

    ComplexBox uses

    .. math::

       g=\sum_k(Q_kQ_k^T-Q_k^TPQ_k-Q_kPQ_k^T),\qquad
       G=2gL-P(2gL),\quad P=LL^T.

    The result here is ``G.T`` because ComplexTorch stores ``M=L.T``.
    """
    q = torch.as_tensor(sequence)
    if q.ndim != 3 or q.shape[-1] != q.shape[-2]:
        raise ValueError("sequence must have shape (lags,n,n)")
    m, single = _row_bases(projection, q.shape[-1])
    q = q.to(dtype=m.dtype, device=m.device)
    l = m.transpose(-1, -2)
    p = l @ l.transpose(-1, -2)
    qqt = torch.sum(q @ q.transpose(-1, -2), dim=0)
    raw = qqt.unsqueeze(0) @ l
    for qk in q:
        raw = raw - qk.transpose(-1, -2) @ p @ qk @ l
        raw = raw - qk @ p @ qk.transpose(-1, -2) @ l
    grad_l = 2.0 * raw
    grad_l = grad_l - p @ grad_l
    grad_m = grad_l.transpose(-1, -2)
    magnitude = torch.linalg.vector_norm(grad_m, dim=(-2, -1))
    if single:
        return grad_m[0], magnitude[0]
    return grad_m, magnitude


def spectral_dynamical_dependence(
    projection: torch.Tensor,
    transfer: torch.Tensor,
) -> torch.Tensor:
    r"""Evaluate ComplexBox ``trfun2dd`` for identity innovations.

    ``transfer`` has shape ``(frequencies,n,n)``. The one-sided trapezoidal
    normalisation is exactly the convention used by ComplexBox/SSDI.
    """
    h = torch.as_tensor(transfer)
    if h.ndim != 3 or h.shape[-1] != h.shape[-2] or h.shape[0] < 2:
        raise ValueError(
            "transfer must have shape (frequencies,n,n), frequencies>=2"
        )
    m, single = _row_bases(projection, h.shape[-1])
    real_dtype = m.dtype
    complex_dtype = (
        torch.complex128 if real_dtype == torch.float64 else torch.complex64
    )
    h = h.to(dtype=complex_dtype, device=m.device)
    l = m.transpose(-1, -2).to(complex_dtype)
    hl = torch.einsum("fij,bjm->bfim", h.conj().transpose(-1, -2), l)
    gram = torch.einsum("bfim,bfin->bfmn", hl.conj(), hl)
    gram = 0.5 * (gram + gram.conj().transpose(-1, -2))
    chol = torch.linalg.cholesky(gram)
    diagonal = torch.real(torch.diagonal(chol, dim1=-2, dim2=-1))
    logdet = 2.0 * torch.sum(torch.log(diagonal), dim=-1)
    weights = torch.ones(h.shape[0], dtype=real_dtype, device=m.device)
    weights[0] = 0.5
    weights[-1] = 0.5
    weights = weights / float(h.shape[0] - 1)
    value = torch.sum(logdet * weights.unsqueeze(0), dim=-1)
    return value[0] if single else value


def spectral_dynamical_dependence_gradient(
    projection: torch.Tensor,
    transfer: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Return ComplexBox/SSDI's analytic spectral Grassmannian gradient.

    This ports Appendix-D/``trfun2ddgrad`` in the row-projection convention.
    Cholesky solves replace explicit inverses without changing the equation.
    """
    h = torch.as_tensor(transfer)
    if h.ndim != 3 or h.shape[-1] != h.shape[-2] or h.shape[0] < 2:
        raise ValueError(
            "transfer must have shape (frequencies,n,n), frequencies>=2"
        )
    m, single = _row_bases(projection, h.shape[-1])
    real_dtype = m.dtype
    complex_dtype = (
        torch.complex128 if real_dtype == torch.float64 else torch.complex64
    )
    h = h.to(dtype=complex_dtype, device=m.device)
    l = m.transpose(-1, -2).to(complex_dtype)
    weights = torch.ones(h.shape[0], dtype=real_dtype, device=m.device)
    weights[0] = 0.5
    weights[-1] = 0.5
    weights = weights / float(h.shape[0] - 1)
    grad_l = torch.zeros_like(l.real)
    for index, hk in enumerate(h):
        hl = hk.conj().transpose(-1, -2).unsqueeze(0) @ l
        gram = hl.conj().transpose(-1, -2) @ hl
        gram = 0.5 * (gram + gram.conj().transpose(-1, -2))
        chol = torch.linalg.cholesky(gram)
        numerator = hk.unsqueeze(0) @ hl
        solved = torch.cholesky_solve(
            numerator.conj().transpose(-1, -2), chol
        )
        right_divided = solved.conj().transpose(-1, -2)
        grad_l = grad_l + 2.0 * weights[index] * right_divided.real
    grad_l = grad_l - 2.0 * l.real
    grad_m = grad_l.transpose(-1, -2)
    magnitude = torch.linalg.vector_norm(grad_m, dim=(-2, -1))
    if single:
        return grad_m[0], magnitude[0]
    return grad_m, magnitude

