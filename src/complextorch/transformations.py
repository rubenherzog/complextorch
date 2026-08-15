r"""Canonical transformations of fitted linear Gaussian systems.

These transformations act on supplied generative models and never refit
observations.  All model families are normalized to the common innovations
representation before the transformation is applied.
"""
from __future__ import annotations

from typing import Literal

import torch

from .control import (
    InnovationsStateSpace,
    _as_innovations_state_space,
    solve_generalized_dare,
)
from .linalg import spd_solve, symmetrise
from .representations import StateSpaceModel, VARSystem


ModelSystem = StateSpaceModel | InnovationsStateSpace | VARSystem


def as_innovations_state_space(system: ModelSystem) -> InnovationsStateSpace:
    """Return a canonical system in exact steady-state innovations form.

    Parameters
    ----------
    system
        Canonical VAR, general state-space, or innovations-form system.

    Returns
    -------
    InnovationsStateSpace
        Exactly equivalent observation process represented as
        ``x[t+1] = A x[t] + K e[t]`` and ``y[t] = C x[t] + e[t]``.

    Notes
    -----
    ``InnovationsStateSpace`` inputs are returned unchanged. ``VARSystem``
    inputs use their exact companion innovations representation, while
    ``StateSpaceModel`` inputs use the steady-state DARE conversion.
    """
    return _as_innovations_state_space(system)


def _batched_matrix(value: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Normalize a matrix to one leading batch dimension."""
    tensor = torch.as_tensor(value)
    single = tensor.ndim == 2
    if single:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError("system matrices must be unbatched or singly batched")
    return tensor, single


def scale_dynamics(
    system: ModelSystem,
    lambda_: float | torch.Tensor,
    *,
    backend: Literal["scipy", "torch"] = "torch",
    rtol: float | None = None,
    atol: float | None = None,
    max_iter: int = 100,
) -> InnovationsStateSpace:
    r"""Scale latent dynamical strength while preserving stochastic structure.

    The input is first converted exactly to innovations form

    .. math::

       x_{t+1}=Ax_t+K\varepsilon_t,\qquad
       y_t=Cx_t+\varepsilon_t,\qquad
       \operatorname{cov}(\varepsilon_t)=V.

    This defines the equivalent correlated-noise generative model

    .. math::

       Q=KVK^\top,\qquad R=V,\qquad S=KV.

    Only the transition is changed,
    :math:`A_\lambda=\lambda A`.  The observation map and stochastic
    architecture :math:`C,Q,R,S` remain fixed.  The generalized steady-state
    DARE is then solved to return the exact innovations representation of the
    scaled process.  Consequently

    .. math::

       \rho(A_\lambda)=\lambda\rho(A)

    for non-negative :math:`\lambda`.  ``lambda_=1`` recovers the supplied
    empirical process up to numerical Riccati tolerance, while ``lambda_=0``
    removes latent temporal propagation.

    Parameters
    ----------
    system
        ``VARSystem``, ``StateSpaceModel``, or ``InnovationsStateSpace``.
    lambda_
        Non-negative scalar or one-dimensional common grid. A scalar is
        applied to every input batch element. A grid is applied to every input
        system and the Cartesian product is flattened in system-major order:
        ``B`` systems and ``L`` values return a batch of ``B * L`` systems.
    backend, rtol, atol, max_iter
        Numerical options forwarded to :func:`complextorch.solve_generalized_dare`.

    Returns
    -------
    InnovationsStateSpace
        Exact innovations representation of the scaled, strictly stable
        process.

    Raises
    ------
    ValueError
        If ``lambda_`` is invalid or any scaled transition is not strictly
        stable.

    Notes
    -----
    For a stable input with nonzero spectral radius, the stability boundary is
    :math:`\lambda_c=1/\rho(A)`.  The boundary itself is excluded because
    stationary covariance-based measures require strict stability.  This is an
    analytical model transformation, not a refit to observations.
    """
    innovations = as_innovations_state_space(system)
    a, a_single = _batched_matrix(innovations.transition)
    c, c_single = _batched_matrix(innovations.observation)
    k, k_single = _batched_matrix(innovations.gain)
    v, v_single = _batched_matrix(innovations.innovation_covariance)

    if not all(tensor.is_floating_point() for tensor in (a, c, k, v)):
        raise TypeError("system matrices must use floating-point dtypes")
    dtypes = {tensor.dtype for tensor in (a, c, k, v)}
    devices = {tensor.device for tensor in (a, c, k, v)}
    if len(dtypes) != 1 or len(devices) != 1:
        raise ValueError("system matrices must share dtype and device")

    batch = max(tensor.shape[0] for tensor in (a, c, k, v))
    if any(tensor.shape[0] not in (1, batch) for tensor in (a, c, k, v)):
        raise ValueError("incompatible system batch dimensions")
    a, c, k, v = [
        tensor.expand(batch, *tensor.shape[1:])
        if tensor.shape[0] == 1
        else tensor
        for tensor in (a, c, k, v)
    ]

    scale = torch.as_tensor(lambda_, dtype=a.dtype, device=a.device)
    if scale.ndim > 1:
        raise ValueError("lambda_ must be a scalar or one-dimensional grid")
    if not bool(torch.isfinite(scale).all().item()):
        raise ValueError("lambda_ must contain only finite values")
    if bool(torch.any(scale < 0).item()):
        raise ValueError("lambda_ must be non-negative")

    scalar_scale = scale.ndim == 0
    if scalar_scale:
        scale = scale.reshape(1)
    elif scale.numel() == 0:
        raise ValueError("lambda_ grid must contain at least one value")

    n_scale = int(scale.numel())

    def expand_grid(tensor: torch.Tensor) -> torch.Tensor:
        """Broadcast one batched system matrix over the common lambda grid."""
        return tensor[:, None].expand(
            batch, n_scale, *tensor.shape[1:]
        ).reshape(batch * n_scale, *tensor.shape[1:])

    a, c, k, v = [expand_grid(tensor) for tensor in (a, c, k, v)]
    scale_flat = scale.repeat(batch)
    scaled_a = a * scale_flat[:, None, None]

    radius = torch.linalg.eigvals(scaled_a).abs().amax(-1)
    if bool(torch.any(radius >= 1).item()):
        raise ValueError(
            "scaled dynamics must be strictly stable; require "
            "lambda_ * spectral_radius < 1"
        )

    process_covariance = k @ v @ k.transpose(-1, -2)
    observation_covariance = v
    cross_covariance = k @ v
    prediction_covariance = solve_generalized_dare(
        scaled_a,
        c,
        process_covariance,
        observation_covariance,
        cross_covariance,
        backend=backend,
        rtol=rtol,
        atol=atol,
        max_iter=max_iter,
    )
    if prediction_covariance.ndim == 2:
        prediction_covariance = prediction_covariance.unsqueeze(0)

    scaled_v = symmetrise(
        c @ prediction_covariance @ c.transpose(-1, -2)
        + observation_covariance
    )
    numerator = (
        scaled_a @ prediction_covariance @ c.transpose(-1, -2)
        + cross_covariance
    )
    identity = torch.eye(
        scaled_v.shape[-1], dtype=scaled_v.dtype, device=scaled_v.device
    ).expand_as(scaled_v)
    scaled_k = numerator @ spd_solve(scaled_v, identity)

    original_single = a_single and c_single and k_single and v_single
    if original_single and scalar_scale:
        return InnovationsStateSpace(
            scaled_a[0], c[0], scaled_k[0], scaled_v[0]
        )
    return InnovationsStateSpace(scaled_a, c, scaled_k, scaled_v)


__all__ = ["ModelSystem", "as_innovations_state_space", "scale_dynamics"]
