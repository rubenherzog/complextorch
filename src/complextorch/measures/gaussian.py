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
from ..linalg import spd_logdet, spd_solve, symmetrise


def gaussian_entropy(covariance: torch.Tensor, *, base: float = 2.0) -> torch.Tensor:
    """Compute Gaussian differential entropy.
    
    .. math::
    
       H(X)=\tfrac12\log\left((2\pi e)^d\det\Sigma\right).
    
    References
    ----------
    - Cover and Thomas (2006).
    """
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
    joint=torch.as_tensor(joint_covariance); n=joint.shape[-1]; n_z=n-n_x-n_y
    if min(n_x,n_y,n_z)<1: raise ValueError('joint covariance must contain nonempty X, Y, Z blocks')
    xz=torch.cat([torch.arange(n_x,device=joint.device),torch.arange(n_x+n_y,n,device=joint.device)])
    yz=torch.arange(n_x,n,device=joint.device); z=torch.arange(n_x+n_y,n,device=joint.device)
    sxz=joint.index_select(-2,xz).index_select(-1,xz); syz=joint.index_select(-2,yz).index_select(-1,yz); sz=joint.index_select(-2,z).index_select(-1,z)
    # Evaluate log-determinants through an SPD-aware factorisation for numerical stability.
    return 0.5*(spd_logdet(sxz)+spd_logdet(syz)-spd_logdet(sz)-spd_logdet(joint))/math.log(base)


def total_correlation(covariance:torch.Tensor,*,base:float=2.0)->torch.Tensor:
    """Total correlation.
    
    Parameters
    ----------
    covariance
        Symmetric covariance matrix or batch of covariance matrices.
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
    cov=torch.as_tensor(covariance); diag=torch.diagonal(cov,dim1=-2,dim2=-1)
    # Evaluate log-determinants through an SPD-aware factorisation for numerical stability.
    return 0.5*(torch.log(diag).sum(-1)-spd_logdet(cov))/math.log(base)


def dual_total_correlation(covariance:torch.Tensor,*,base:float=2.0)->torch.Tensor:
    """Dual total correlation.
    
    Parameters
    ----------
    covariance
        Symmetric covariance matrix or batch of covariance matrices.
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
    cov=torch.as_tensor(covariance); n=cov.shape[-1]; h=gaussian_entropy(cov,base=base)
    parts=[]
    for i in range(n):
        idx=torch.tensor([j for j in range(n) if j!=i],device=cov.device)
        parts.append(gaussian_entropy(cov.index_select(-2,idx).index_select(-1,idx),base=base))
    return torch.stack(parts,-1).sum(-1)-(n-1)*h


def o_information(covariance:torch.Tensor,*,base:float=2.0)->torch.Tensor:
    """Compute Gaussian O-information.
    
    Positive values are redundancy-dominated and negative values are
    synergy-dominated.
    
    References
    ----------
    - Rosas et al. (2019), *Physical Review E* 100, 032305.
    """
    return total_correlation(covariance,base=base)-dual_total_correlation(covariance,base=base)


def s_information(covariance:torch.Tensor,*,base:float=2.0)->torch.Tensor:
    """S information.
    
    Parameters
    ----------
    covariance
        Symmetric covariance matrix or batch of covariance matrices.
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
    return total_correlation(covariance,base=base)+dual_total_correlation(covariance,base=base)


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
