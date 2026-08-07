"""Gaussian information-theoretic measures from covariance matrices.

For :math:`X\in\mathbb R^d` with covariance :math:`\Sigma`,

.. math::

   H(X)=\tfrac12\log\left((2\pi e)^d\det\Sigma\right).

Mutual information and higher-order quantities are evaluated through stable
log-determinant identities.

References
----------
- Cover, T. M. and Thomas, J. A. (2006). *Elements of Information Theory*.
- Rosas, F. E. et al. (2019). Quantifying high-order interdependencies via the
  O-information. *Physical Review E*, 100, 032305.
"""
from __future__ import annotations
import math
import torch
from ..linalg import spd_logdet, spd_solve, stable_cholesky, symmetrise


def gaussian_entropy(covariance: torch.Tensor, *, base: float = 2.0) -> torch.Tensor:
    """Compute Gaussian differential entropy.
    
    .. math::
    
       H(X)=\tfrac12\log\left((2\pi e)^d\det\Sigma\right).
    
    References
    ----------
    - Cover and Thomas (2006).
    """
    # Evaluate H(X)=1/2[k log(2 pi e)+log det Sigma] for the selected Gaussian covariance.
    cov=torch.as_tensor(covariance)
    d=cov.shape[-1]
    # Evaluate log-determinants through an SPD-aware factorisation for numerical stability.
    value=0.5*(d*math.log(2*math.pi*math.e)+spd_logdet(cov))
    return value/math.log(base)


def conditional_covariance(joint_covariance: torch.Tensor, n_left: int) -> torch.Tensor:
    """Compute a Gaussian conditional covariance by Schur complement.
    
    .. math::
    
       \Sigma_{X\mid Y}
       =\Sigma_{XX}-\Sigma_{XY}\Sigma_{YY}^{-1}\Sigma_{YX}.
    
    References
    ----------
    - Cover and Thomas (2006).
    """
    joint=torch.as_tensor(joint_covariance); n=joint.shape[-1]
    if not 0<n_left<n: raise ValueError('n_left must split the covariance')
    a=joint[...,:n_left,:n_left]; b=joint[...,:n_left,n_left:]; c=joint[...,n_left:,n_left:]
    return symmetrise(a-b@spd_solve(c,b.transpose(-1,-2)))


def gaussian_mutual_information(joint_covariance: torch.Tensor,n_left:int,*,base:float=2.0)->torch.Tensor:
    """Compute Gaussian mutual information from covariance blocks.
    
    .. math::
    
       I(X;Y)=\tfrac12\log
       \frac{\det\Sigma_X\det\Sigma_Y}{\det\Sigma_{XY}}.
    
    References
    ----------
    - Cover and Thomas (2006).
    """
    # Use I(X;Y)=1/2 log(det Sigma_X det Sigma_Y / det Sigma_XY).
    joint=torch.as_tensor(joint_covariance); n=joint.shape[-1]
    if not 0<n_left<n: raise ValueError('n_left must split the covariance')
    a=joint[...,:n_left,:n_left]; b=joint[...,n_left:,n_left:]
    # Evaluate log-determinants through an SPD-aware factorisation for numerical stability.
    return 0.5*(spd_logdet(a)+spd_logdet(b)-spd_logdet(joint))/math.log(base)


def gaussian_conditional_mutual_information(joint_covariance:torch.Tensor,n_x:int,n_y:int,*,base:float=2.0)->torch.Tensor:
    """Gaussian conditional mutual information.
    
    Parameters
    ----------
    joint_covariance
        Input required by this calculation.
    n_x
        Input required by this calculation.
    n_y
        Input required by this calculation.
    base
        Logarithm base used for information quantities.
    
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
    # Use the Gaussian Schur-complement identity for I(X;Y|Z).
    joint=torch.as_tensor(joint_covariance); n=joint.shape[-1]; n_z=n-n_x-n_y
    if min(n_x,n_y,n_z)<1: raise ValueError('joint covariance must contain nonempty X, Y, Z blocks')
    xz=torch.cat([torch.arange(n_x,device=joint.device),torch.arange(n_x+n_y,n,device=joint.device)])
    yz=torch.arange(n_x,n,device=joint.device); z=torch.arange(n_x+n_y,n,device=joint.device)
    sxz=joint.index_select(-2,xz).index_select(-1,xz); syz=joint.index_select(-2,yz).index_select(-1,yz); sz=joint.index_select(-2,z).index_select(-1,z)
    # Evaluate log-determinants through an SPD-aware factorisation for numerical stability.
    return 0.5*(spd_logdet(sxz)+spd_logdet(syz)-spd_logdet(sz)-spd_logdet(joint))/math.log(base)


def _validate_gaussian_covariance(covariance: torch.Tensor, base: float) -> torch.Tensor:
    """Validate a covariance tensor used by Gaussian multivariate measures."""
    cov = torch.as_tensor(covariance)
    if cov.ndim < 2 or cov.shape[-2] != cov.shape[-1]:
        raise ValueError("covariance must have shape (..., n, n)")
    if cov.shape[-1] < 1:
        raise ValueError("covariance must contain at least one variable")
    if not cov.is_floating_point():
        raise TypeError("covariance must use a floating-point dtype")
    if not bool(torch.isfinite(cov).all()):
        raise ValueError("covariance must contain only finite values")
    if bool((torch.diagonal(cov, dim1=-2, dim2=-1) <= 0).any()):
        raise ValueError("covariance diagonal must be strictly positive")
    if not math.isfinite(base) or base <= 0.0 or base == 1.0:
        raise ValueError("base must be finite, positive, and different from 1")
    return cov


def _gaussian_cholesky_terms(
    covariance: torch.Tensor, *, base: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Return validated covariance, Cholesky factor, logdet and log-base scale."""
    cov = _validate_gaussian_covariance(covariance, base)
    chol, _ = stable_cholesky(cov)
    chol_diagonal = torch.diagonal(chol, dim1=-2, dim2=-1)
    logdet = 2.0 * torch.log(chol_diagonal).sum(dim=-1)
    return cov, chol, logdet, math.log(base)


def _precision_diagonal_from_cholesky(chol: torch.Tensor) -> torch.Tensor:
    r"""Return ``diag(Sigma^-1)`` from ``Sigma = L L^T`` without inversion."""
    n = chol.shape[-1]
    eye = torch.eye(n, dtype=chol.dtype, device=chol.device)
    eye = eye.expand(*chol.shape[:-2], n, n)
    inverse_chol = torch.linalg.solve_triangular(chol, eye, upper=False)
    return inverse_chol.square().sum(dim=-2)


def _gaussian_total_correlations(
    covariance: torch.Tensor, *, base: float
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Return Gaussian TC and DTC from one shared Cholesky factorisation.

    For covariance :math:`\Sigma` and precision :math:`\Lambda=\Sigma^{-1}`,

    .. math::

       \mathrm{TC}
       &= \frac{1}{2}\left(\sum_i\log\Sigma_{ii}
          -\log\det\Sigma\right),\\
       \mathrm{DTC}
       &= \frac{1}{2}\left(\log\det\Sigma
          +\sum_i\log\Lambda_{ii}\right).

    The implementation never forms :math:`\Sigma^{-1}`. If
    :math:`\Sigma=LL^\top`, then
    :math:`\Lambda_{ii}=\sum_k(L^{-1})_{ki}^2`; ``L^{-1}`` is obtained by a
    batched triangular solve against the identity. All leading batch
    dimensions, dtype and device are preserved.
    """
    cov, chol, logdet, scale = _gaussian_cholesky_terms(covariance, base=base)
    marginal_variances = torch.diagonal(cov, dim1=-2, dim2=-1)
    tc_nats = 0.5 * (torch.log(marginal_variances).sum(dim=-1) - logdet)
    precision_diagonal = _precision_diagonal_from_cholesky(chol)
    dtc_nats = 0.5 * (logdet + torch.log(precision_diagonal).sum(dim=-1))
    return tc_nats / scale, dtc_nats / scale


def total_correlation(covariance:torch.Tensor,*,base:float=2.0)->torch.Tensor:
    r"""Compute Gaussian total correlation from covariance matrices.

    .. math::

       \mathrm{TC}(X_1,\ldots,X_N)
       =\frac{1}{2\log b}\left[
       \sum_i\log\Sigma_{ii}-\log\det\Sigma\right].

    Parameters
    ----------
    covariance
        Symmetric positive-definite covariance matrix with shape ``(..., N, N)``.
        Any leading batch dimensions are preserved.
    base
        Logarithm base ``b``. Defaults to 2 (bits).

    Returns
    -------
    torch.Tensor
        Total correlation with shape ``covariance.shape[:-2]``.

    Notes
    -----
    The calculation is fully Torch-native and differentiable. It uses one
    Cholesky factorisation and does not construct marginal covariance blocks or
    a precision matrix.
    """
    cov, _, logdet, scale = _gaussian_cholesky_terms(covariance, base=base)
    marginal_variances = torch.diagonal(cov, dim1=-2, dim2=-1)
    return 0.5 * (torch.log(marginal_variances).sum(dim=-1) - logdet) / scale


def dual_total_correlation(covariance:torch.Tensor,*,base:float=2.0)->torch.Tensor:
    r"""Compute Gaussian dual total correlation from covariance matrices.

    .. math::

       \mathrm{DTC}(X_1,\ldots,X_N)
       =\frac{1}{2\log b}\left[
       \log\det\Sigma+\sum_i\log(\Sigma^{-1})_{ii}\right].

    Parameters
    ----------
    covariance
        Symmetric positive-definite covariance matrix with shape ``(..., N, N)``.
        Any leading batch dimensions are preserved.
    base
        Logarithm base ``b``. Defaults to 2 (bits).

    Returns
    -------
    torch.Tensor
        Dual total correlation with shape ``covariance.shape[:-2]``.

    Notes
    -----
    The precision matrix is never formed explicitly. Its diagonal is obtained
    from a batched triangular solve using the Cholesky factor of the covariance.
    """
    _, chol, logdet, scale = _gaussian_cholesky_terms(covariance, base=base)
    precision_diagonal = _precision_diagonal_from_cholesky(chol)
    return 0.5 * (logdet + torch.log(precision_diagonal).sum(dim=-1)) / scale


def o_information(covariance:torch.Tensor,*,base:float=2.0)->torch.Tensor:
    """Compute Gaussian O-information.
    
    Positive values are redundancy-dominated and negative values are
    synergy-dominated.
    
    References
    ----------
    - Rosas et al. (2019), *Physical Review E* 100, 032305.
    """
    tc, dtc = _gaussian_total_correlations(covariance, base=base)
    return tc - dtc


def s_information(covariance:torch.Tensor,*,base:float=2.0)->torch.Tensor:
    """Compute Gaussian S-information as ``TC + DTC``.

    Parameters
    ----------
    covariance
        Symmetric positive-definite covariance matrix with shape ``(..., N, N)``.
    base
        Logarithm base used for information quantities.

    Returns
    -------
    torch.Tensor
        S-information with all leading batch dimensions preserved.
    """
    tc, dtc = _gaussian_total_correlations(covariance, base=base)
    return tc + dtc


def local_gaussian_mutual_information(samples:torch.Tensor,joint_covariance:torch.Tensor,n_left:int,*,mean:torch.Tensor|None=None,base:float=2.0)->torch.Tensor:
    """Local gaussian mutual information.
    
    Parameters
    ----------
    samples
        Input required by this calculation.
    joint_covariance
        Input required by this calculation.
    n_left
        Input required by this calculation.
    mean
        Input required by this calculation.
    base
        Logarithm base used for information quantities.
    
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
    x=torch.as_tensor(samples); cov=torch.as_tensor(joint_covariance,dtype=x.dtype,device=x.device); n=cov.shape[-1]
    if x.shape[-1]!=n or not 0<n_left<n: raise ValueError('invalid sample or split shape')
    mu=torch.zeros(n,dtype=x.dtype,device=x.device) if mean is None else torch.as_tensor(mean,dtype=x.dtype,device=x.device)
    z=x-mu; left=z[...,:n_left]; right=z[...,n_left:]
    a=cov[...,:n_left,:n_left]; b=cov[...,n_left:,n_left:]
    q_joint=(z.unsqueeze(-2)@spd_solve(cov,z.unsqueeze(-1))).squeeze(-1).squeeze(-1)
    q_left=(left.unsqueeze(-2)@spd_solve(a,left.unsqueeze(-1))).squeeze(-1).squeeze(-1)
    q_right=(right.unsqueeze(-2)@spd_solve(b,right.unsqueeze(-1))).squeeze(-1).squeeze(-1)
    # Evaluate log-determinants through an SPD-aware factorisation for numerical stability.
    logdet_term=0.5*(spd_logdet(a)+spd_logdet(b)-spd_logdet(cov))
    return (logdet_term-0.5*(q_joint-q_left-q_right))/math.log(base)
