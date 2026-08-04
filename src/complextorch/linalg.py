"""Numerically explicit batched linear-algebra primitives."""
from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class LyapunovInfo:
    method: str
    iterations: int
    converged: bool
    residual_max: float

def symmetrise(x: torch.Tensor) -> torch.Tensor:
    return 0.5*(x+x.transpose(-1,-2))

def stable_cholesky(matrix: torch.Tensor, *, jitter: float=1e-10, max_tries: int=6):
    if matrix.shape[-1]!=matrix.shape[-2]: raise ValueError("matrix must be square")
    matrix=symmetrise(matrix); n=matrix.shape[-1]; batch_shape=matrix.shape[:-2]
    eye=torch.eye(n,dtype=matrix.dtype,device=matrix.device).expand(*batch_shape,n,n)
    used=torch.zeros(batch_shape,dtype=matrix.dtype,device=matrix.device); candidate=matrix
    for attempt in range(max_tries+1):
        chol,info=torch.linalg.cholesky_ex(candidate,check_errors=False); good=info==0
        if bool(good.all()): return chol,used
        if attempt==max_tries: raise torch.linalg.LinAlgError(f"Cholesky failed for {int((~good).sum())} batch item(s)")
        add=jitter*(10.0**attempt); used=torch.where(good,used,torch.full_like(used,add)); candidate=matrix+used[...,None,None]*eye
    raise RuntimeError("unreachable")

def spd_logdet(matrix: torch.Tensor, *, jitter: float=1e-10) -> torch.Tensor:
    chol,_=stable_cholesky(matrix,jitter=jitter); return 2.0*torch.log(torch.diagonal(chol,dim1=-2,dim2=-1)).sum(-1)

def spd_solve(matrix: torch.Tensor, rhs: torch.Tensor, *, jitter: float=1e-10) -> torch.Tensor:
    chol,_=stable_cholesky(matrix,jitter=jitter); return torch.cholesky_solve(rhs,chol)

def spectral_radius(matrix: torch.Tensor) -> torch.Tensor:
    return torch.linalg.eigvals(matrix).abs().amax(-1)

def solve_discrete_lyapunov(transition: torch.Tensor, noise_cov: torch.Tensor, *, method: str="doubling", rtol: float=1e-10, atol: float=1e-12, max_iter: int=100, check_stability: bool=True):
    a=transition; q=noise_cov; unbatched=a.ndim==2
    if unbatched: a=a.unsqueeze(0); q=q.unsqueeze(0)
    if a.ndim!=3 or q.ndim!=3 or a.shape!=q.shape: raise ValueError("transition and noise_cov must have shape (batch,n,n)")
    if check_stability and bool((spectral_radius(a)>=1).any()): raise ValueError("discrete Lyapunov requires stable A")
    if method=="direct":
        batch,n,_=a.shape; eye=torch.eye(n*n,dtype=a.dtype,device=a.device).expand(batch,n*n,n*n)
        kron=torch.einsum("bij,bkl->bikjl",a,a).reshape(batch,n*n,n*n)
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
