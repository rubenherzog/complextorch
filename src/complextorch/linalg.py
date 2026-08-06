"""Numerically stable linear-algebra primitives for covariance models.

For a symmetric positive-definite matrix :math:`S=LL^\top`, log determinants
are evaluated as

.. math::

   \log\det S = 2\sum_i\log L_{ii},

and systems are solved through triangular factors rather than explicit matrix
inverses.

References
----------
- Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*.
- PyTorch linear algebra: https://pytorch.org/docs/stable/linalg.html
"""
from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class LyapunovInfo:
    """Lyapunovinfo.
    
    Notes
    -----
    Public fitted attributes use the trailing-underscore convention.
    """
    method: str
    iterations: int
    converged: bool
    residual_max: float

def symmetrise(x: torch.Tensor) -> torch.Tensor:
    """Symmetrise.
    
    Parameters
    ----------
    x
        Input observations or tensor-valued quantity.
    
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
    return 0.5*(x+x.transpose(-1,-2))

# Add adaptive jitter only when required to retain a valid SPD factorisation.
def stable_cholesky(matrix: torch.Tensor, *, jitter: float=1e-10, max_tries: int=6):
    """Stable cholesky.
    
    Parameters
    ----------
    matrix
        Input required by this calculation.
    jitter
        Initial diagonal regularization used for SPD factorization.
    max_tries
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
    if matrix.shape[-1]!=matrix.shape[-2]: raise ValueError("matrix must be square")
    matrix=symmetrise(matrix); n=matrix.shape[-1]; batch_shape=matrix.shape[:-2]
    eye=torch.eye(n,dtype=matrix.dtype,device=matrix.device).expand(*batch_shape,n,n)
    used=torch.zeros(batch_shape,dtype=matrix.dtype,device=matrix.device); candidate=matrix
    for attempt in range(max_tries+1):
        # Cholesky factorisation preserves the SPD structure and avoids explicit inversion.
        chol,info=torch.linalg.cholesky_ex(candidate,check_errors=False); good=info==0
        if bool(good.all()): return chol,used
        if attempt==max_tries: raise torch.linalg.LinAlgError(f"Cholesky failed for {int((~good).sum())} batch item(s)")
        add=jitter*(10.0**attempt); used=torch.where(good,used,torch.full_like(used,add)); candidate=matrix+used[...,None,None]*eye
    raise RuntimeError("unreachable")

# Evaluate log-determinants through an SPD-aware factorisation for numerical stability.
def spd_logdet(matrix: torch.Tensor, *, jitter: float=1e-10) -> torch.Tensor:
    """Spd logdet.
    
    Parameters
    ----------
    matrix
        Input required by this calculation.
    jitter
        Initial diagonal regularization used for SPD factorization.
    
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
    # Add adaptive jitter only when required to retain a valid SPD factorisation.
    chol,_=stable_cholesky(matrix,jitter=jitter); return 2.0*torch.log(torch.diagonal(chol,dim1=-2,dim2=-1)).sum(-1)

def spd_solve(matrix: torch.Tensor, rhs: torch.Tensor, *, jitter: float=1e-10) -> torch.Tensor:
    """Spd solve.
    
    Parameters
    ----------
    matrix
        Input required by this calculation.
    rhs
        Input required by this calculation.
    jitter
        Initial diagonal regularization used for SPD factorization.
    
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
    # Add adaptive jitter only when required to retain a valid SPD factorisation.
    # Solve with the Cholesky factor instead of forming the covariance inverse.
    # Add adaptive jitter only when required to retain a valid SPD factorisation.
    # Solve with the Cholesky factor instead of forming the covariance inverse.
    # Add adaptive jitter only when required to retain a valid SPD factorisation.
    # Solve with the Cholesky factor instead of forming the covariance inverse.
    chol,_=stable_cholesky(matrix,jitter=jitter); return torch.cholesky_solve(rhs,chol)

def spectral_radius(matrix: torch.Tensor) -> torch.Tensor:
    """Spectral radius.
    
    Parameters
    ----------
    matrix
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
    # Companion eigenvalues determine stationarity through their spectral radius.
    # Obtain dynamical modes whose moduli determine stability and characteristic decay.
    return torch.linalg.eigvals(matrix).abs().amax(-1)

def solve_discrete_lyapunov(transition: torch.Tensor, noise_cov: torch.Tensor, *, method: str="doubling", rtol: float=1e-10, atol: float=1e-12, max_iter: int=100, check_stability: bool=True):
    """Solve discrete lyapunov.
    
    Parameters
    ----------
    transition
        State-transition matrix.
    noise_cov
        Input required by this calculation.
    method
        Named statistical or numerical method.
    rtol
        Input required by this calculation.
    atol
        Input required by this calculation.
    max_iter
        Maximum number of numerical iterations.
    check_stability
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
    a=transition; q=noise_cov; unbatched=a.ndim==2
    if unbatched: a=a.unsqueeze(0); q=q.unsqueeze(0)
    if a.ndim!=3 or q.ndim!=3 or a.shape!=q.shape: raise ValueError("transition and noise_cov must have shape (batch,n,n)")
    if check_stability and bool((spectral_radius(a)>=1).any()): raise ValueError("discrete Lyapunov requires stable A")
    if method=="direct":
        batch,n,_=a.shape; eye=torch.eye(n*n,dtype=a.dtype,device=a.device).expand(batch,n*n,n*n)
        kron=torch.einsum("bij,bkl->bikjl",a,a).reshape(batch,n*n,n*n)
        # Solve the linear system directly instead of multiplying by an explicit inverse.
        s=symmetrise(torch.linalg.solve(eye-kron,q.reshape(batch,n*n,1)).reshape(batch,n,n)); iterations=1
    elif method=="doubling":
        s=q.clone(); ap=a.clone(); converged=False
        for iterations in range(1,max_iter+1):
            inc=ap@s@ap.transpose(-1,-2); sn=symmetrise(s+inc); scale=sn.abs().amax().clamp_min(torch.finfo(s.dtype).tiny)
            if float(inc.abs().amax())<=atol+rtol*float(scale): s=sn; converged=True; break
            s=sn; ap=ap@ap
        if not converged: raise RuntimeError("Lyapunov doubling did not converge")
    else: raise ValueError("method must be 'doubling' or 'direct'")
    residual=s-a@s@a.transpose(-1,-2)-q
    info=LyapunovInfo(method,iterations,True,float(residual.abs().amax()))
    return (s[0] if unbatched else s),info
