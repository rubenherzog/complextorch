r"""Simulation of stationary Gaussian VAR systems.

Trajectories obey

.. math::

   x_t=c+\sum_{k=1}^{p}A_kx_{t-k}+\varepsilon_t.

Automatic burn-in uses the companion spectral radius to reduce the influence
of initial conditions below a requested tolerance.

References
----------
- Lütkepohl, H. (2005), Chapters 2--3.
- ComplexBox: https://github.com/bmilinkovic/complexbox
"""
from __future__ import annotations
import math
import torch
from .linalg import spectral_radius
from .representations import companion_matrix


def _normalise_coefficients(coefficients):
    """Normalise coefficients.
    
    Parameters
    ----------
    coefficients
        VAR coefficient tensor ordered by lag, target and source.
    
    Returns
    -------
    object
        Computed result; see the annotated return type and shape notes.
    
    Notes
    -----
    Batch dimensions are preserved unless explicitly documented otherwise.
    The implementation validates dimensional and positive-definiteness
    requirements before executing the numerical core.
    """
    coef = torch.as_tensor(coefficients)
    if coef.ndim == 3:
        coef = coef.unsqueeze(0)
    if coef.ndim != 4:
        raise ValueError("coefficients must have shape (p,n,n) or (batch,p,n,n)")
    return coef


def _normalise_covariance(covariance, batch, n):
    """Normalise covariance.
    
    Parameters
    ----------
    covariance
        Symmetric covariance matrix or batch of covariance matrices.
    batch
        Input required by this calculation.
    n
        Input required by this calculation.
    
    Returns
    -------
    object
        Computed result; see the annotated return type and shape notes.
    
    Notes
    -----
    Batch dimensions are preserved unless explicitly documented otherwise.
    The implementation validates dimensional and positive-definiteness
    requirements before executing the numerical core.
    """
    q = torch.as_tensor(covariance)
    if q.ndim == 2:
        q = q.unsqueeze(0)
    if q.shape[0] == 1 and batch > 1:
        q = q.expand(batch, -1, -1)
    if q.shape != (batch, n, n):
        raise ValueError("covariance has incompatible shape")
    return q


def automatic_burnin(coefficients, *, epsilon: float | None = None) -> int:
    r"""Choose burn-in length from the companion spectral radius.

    The returned value is the smallest integer :math:`T` satisfying
    :math:`\rho^T < \epsilon`, where :math:`\rho` is the largest companion
    spectral radius in the supplied batch.

    Parameters
    ----------
    coefficients
        VAR coefficient tensor with shape ``(p, n, n)`` or
        ``(batch, p, n, n)``.
    epsilon
        Positive decay tolerance. If omitted, the machine epsilon of the
        coefficient dtype is used.

    Returns
    -------
    int
        Burn-in length shared by the supplied systems.

    References
    ----------
    - Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*.
    - ComplexBox simulation convention.
    """
    coef = _normalise_coefficients(coefficients)
    rho = float(spectral_radius(companion_matrix(coef)).max())
    if rho >= 1:
        return 0
    if rho <= 0:
        return 1
    eps = torch.finfo(coef.dtype).eps if epsilon is None else float(epsilon)
    return int(math.ceil(-math.log(eps) / -math.log(rho)))


def simulate_var(
    coefficients,
    innovation_covariance,
    n_times: int,
    *,
    burnin: int | str = 500,
    seed: int = 0,
    return_innovations: bool = False,
):
    r"""Simulate one or more stationary Gaussian VAR trajectories.

    Samples follow

    .. math::

       x_t = \sum_{k=1}^{p} A_k x_{t-k} + \varepsilon_t,
       \qquad
       \varepsilon_t \sim \mathcal N(0,\Sigma).

    Parameters
    ----------
    coefficients
        Stable VAR coefficients with shape ``(p, n, n)`` or
        ``(batch, p, n, n)``.
    innovation_covariance
        Positive-definite innovations covariance with shape ``(n, n)`` or
        ``(batch, n, n)``.
    n_times
        Number of retained samples per trajectory.
    burnin
        Non-negative number of discarded initial samples, or ``"auto"`` to use
        :func:`automatic_burnin`.
    seed
        Seed for the local Torch random generator.
    return_innovations
        If true, also return the retained innovation sequence.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, torch.Tensor]
        Simulated observations with shape ``(batch, n_times, n)`` and,
        optionally, innovations with the same shape.

    Notes
    -----
    Every batch item is simulated as an independent trajectory. No state or lag
    is propagated across batch boundaries.

    References
    ----------
    - Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*.
    - ComplexBox simulation convention.
    """
    # Generate samples from x_t = sum_k A_k x_(t-k) + epsilon_t.
    coef = _normalise_coefficients(coefficients)
    batch, order, n, _ = coef.shape
    q = _normalise_covariance(innovation_covariance, batch, n).to(coef)
    if bool((spectral_radius(companion_matrix(coef)) >= 1).any()):
        raise ValueError("simulation requires stable coefficients")
    if isinstance(burnin, str):
        if burnin.lower() != "auto":
            raise ValueError("burnin string must be 'auto'")
        burnin_value = automatic_burnin(coef)
    else:
        burnin_value = int(burnin)
        if burnin_value < 0:
            raise ValueError("burnin must be nonnegative")
    generator = torch.Generator(device=coef.device)
    generator.manual_seed(seed)
    # Factor the covariance once; no explicit inverse is formed.
    chol = torch.linalg.cholesky(q)
    total = n_times + burnin_value
    innovations = torch.randn(
        (batch, total, n), dtype=coef.dtype, device=coef.device, generator=generator
    )
    innovations = torch.einsum("bij,btj->bti", chol, innovations)
    x = torch.zeros_like(innovations)
    for time in range(total):
        value = innovations[:, time].clone()
        for lag in range(1, order + 1):
            if time >= lag:
                value = value + torch.einsum(
                    "bij,bj->bi", coef[:, lag - 1], x[:, time - lag]
                )
        x[:, time] = value
    observations = x[:, burnin_value:]
    retained_innovations = innovations[:, burnin_value:]
    return (observations, retained_innovations) if return_innovations else observations


# Synthetic model/covariance generators live in ``synthetic``.
# These re-exports preserve historical module-level imports.
from .synthetic import (
    demo_var,
    random_correlation_matrix,
    random_positive_definite_covariance,
    random_stable_var,
)
