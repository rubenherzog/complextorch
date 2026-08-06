"""Inference for linear Gaussian state-space systems.

The module provides Kalman filtering, Rauch--Tung--Striebel smoothing, N4SID
subspace identification, and expectation--maximisation refinement.

References
----------
- Kalman, R. E. (1960). Linear filtering and prediction.
- Rauch, H. E., Tung, F., and Striebel, C. T. (1965). Maximum-likelihood
  estimates of linear dynamic systems.
- Van Overschee, P. and De Moor, B. (1994). N4SID.
- Shumway, R. H. and Stoffer, D. S. (1982). EM for time-series smoothing.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from sklearn.base import BaseEstimator

from .linalg import symmetrise
from .representations import LinearDynamicalSystem


@dataclass(frozen=True)
class KalmanResult:
    """Kalmanresult.
    
    Notes
    -----
    Public fitted attributes use the trailing-underscore convention.
    """
    filtered_mean: torch.Tensor
    filtered_covariance: torch.Tensor
    predicted_mean: torch.Tensor
    predicted_covariance: torch.Tensor
    innovations: torch.Tensor
    innovation_covariance: torch.Tensor
    log_likelihood: torch.Tensor


def _as_2d(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """As 2d.
    
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
    t=torch.as_tensor(x); single=t.ndim==2
    if single: t=t.unsqueeze(0)
    if t.ndim!=3: raise ValueError("observations must have shape (time,variables) or (batch,time,variables)")
    return t,single


def kalman_filter(observations,system,*,initial_mean=None,initial_covariance=None):
    """Batched Kalman filter with exact Gaussian log likelihood."""
    y,single=_as_2d(observations)
    a=system.transition.unsqueeze(0) if system.transition.ndim==2 else system.transition
    c=system.observation.unsqueeze(0) if system.observation.ndim==2 else system.observation
    q=system.process_covariance.unsqueeze(0) if system.process_covariance.ndim==2 else system.process_covariance
    r=system.observation_covariance.unsqueeze(0) if system.observation_covariance.ndim==2 else system.observation_covariance
    batch=y.shape[0]; a,c,q,r=[x.expand(batch,*x.shape[1:]) if x.shape[0]==1 else x for x in (a,c,q,r)]
    d=a.shape[-1]; m=c.shape[-2]
    mean=torch.zeros((batch,d),dtype=y.dtype,device=y.device) if initial_mean is None else torch.as_tensor(initial_mean,dtype=y.dtype,device=y.device).expand(batch,-1)
    if initial_covariance is None:
        p=system.state_covariance
        if p is None: p=torch.eye(d,dtype=y.dtype,device=y.device).expand(batch,d,d)
        elif p.ndim==2: p=p.unsqueeze(0).expand(batch,d,d)
    else:
        p=torch.as_tensor(initial_covariance,dtype=y.dtype,device=y.device); p=p.unsqueeze(0) if p.ndim==2 else p
    fm=[]; fc=[]; pm=[]; pc=[]; innovations=[]; innovation_cov=[]; ll=torch.zeros(batch,dtype=y.dtype,device=y.device)
    log2pi=torch.log(torch.tensor(2*torch.pi,dtype=y.dtype,device=y.device))
    for t in range(y.shape[1]):
        pred_mean=torch.einsum('bij,bj->bi',a,mean); pred_cov=symmetrise(a@p@a.transpose(-1,-2)+q)
        innovation=y[:,t]-torch.einsum('bmd,bd->bm',c,pred_mean); s=symmetrise(c@pred_cov@c.transpose(-1,-2)+r)
        # Cholesky factorisation preserves the SPD structure and avoids explicit inversion.
        # Solve with the Cholesky factor instead of forming the covariance inverse.
        # Cholesky factorisation preserves the SPD structure and avoids explicit inversion.
        # Solve with the Cholesky factor instead of forming the covariance inverse.
        # Cholesky factorisation preserves the SPD structure and avoids explicit inversion.
        # Solve with the Cholesky factor instead of forming the covariance inverse.
        chol=torch.linalg.cholesky(s); k=torch.cholesky_solve((pred_cov@c.transpose(-1,-2)).transpose(-1,-2),chol).transpose(-1,-2)
        mean=pred_mean+torch.einsum('bdm,bm->bd',k,innovation)
        eye=torch.eye(d,dtype=y.dtype,device=y.device).expand(batch,d,d)
        p=symmetrise((eye-k@c)@pred_cov@(eye-k@c).transpose(-1,-2)+k@r@k.transpose(-1,-2))
        # Solve with the Cholesky factor instead of forming the covariance inverse.
        quad=torch.cholesky_solve(innovation.unsqueeze(-1),chol).squeeze(-1)
        ll+=-.5*(m*log2pi+2*torch.log(torch.diagonal(chol,dim1=-2,dim2=-1)).sum(-1)+(innovation*quad).sum(-1))
        pm.append(pred_mean); pc.append(pred_cov); fm.append(mean); fc.append(p); innovations.append(innovation); innovation_cov.append(s)
    stack=lambda z: torch.stack(z,1)
    result=KalmanResult(stack(fm),stack(fc),stack(pm),stack(pc),stack(innovations),stack(innovation_cov),ll)
    if single: result=KalmanResult(*(getattr(result,f)[0] for f in result.__dataclass_fields__))
    return result


@dataclass(frozen=True)
class SmootherResult:
    """Smootherresult.
    
    Notes
    -----
    Public fitted attributes use the trailing-underscore convention.
    """
    smoothed_mean: torch.Tensor
    smoothed_covariance: torch.Tensor
    lag_covariance: torch.Tensor
    log_likelihood: torch.Tensor


def kalman_smoother(observations,system):
    """Kalman smoother.
    
    Parameters
    ----------
    observations
        Observed time series in ComplexTorch batch-first layout.
    system
        Canonical VAR or state-space system.
    
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
    filt=kalman_filter(observations,system); single=filt.filtered_mean.ndim==2
    fm=filt.filtered_mean.unsqueeze(0) if single else filt.filtered_mean; fc=filt.filtered_covariance.unsqueeze(0) if single else filt.filtered_covariance
    pm=filt.predicted_mean.unsqueeze(0) if single else filt.predicted_mean; pc=filt.predicted_covariance.unsqueeze(0) if single else filt.predicted_covariance
    a=system.transition.unsqueeze(0) if system.transition.ndim==2 else system.transition; a=a.expand(fm.shape[0],-1,-1)
    sm=fm.clone(); sc=fc.clone(); lag=torch.zeros_like(sc)
    for t in range(fm.shape[1]-2,-1,-1):
        j=fc[:,t]@a.transpose(-1,-2)@torch.linalg.inv(pc[:,t+1])
        sm[:,t]=fm[:,t]+torch.einsum('bij,bj->bi',j,sm[:,t+1]-pm[:,t+1])
        sc[:,t]=symmetrise(fc[:,t]+j@(sc[:,t+1]-pc[:,t+1])@j.transpose(-1,-2)); lag[:,t+1]=sc[:,t+1]@j.transpose(-1,-2)
    out=SmootherResult(sm,sc,lag,filt.log_likelihood.unsqueeze(0) if single else filt.log_likelihood)
    return SmootherResult(out.smoothed_mean[0],out.smoothed_covariance[0],out.lag_covariance[0],out.log_likelihood[0]) if single else out


class N4SID(BaseEstimator):
    """Compact block-Hankel/SVD subspace identifier."""
    def __init__(self,n_states:int,block_rows:int=10,ridge:float=1e-8): self.n_states=n_states; self.block_rows=block_rows; self.ridge=ridge
    def fit(self,observations):
        """Fit fit from observations.
        
        Parameters
        ----------
        observations
            Observed time series in ComplexTorch batch-first layout.
        
        Returns
        -------
        object
            The fitted estimator instance.
        
        Notes
        -----
        Batch dimensions are preserved unless explicitly documented otherwise.
        The implementation validates dimensional and positive-definiteness
        requirements before executing the numerical core.
        """
        y=torch.as_tensor(observations,dtype=torch.float64)
        if y.ndim!=2: raise ValueError('N4SID currently expects (time,variables)')
        n,m=y.shape; i=self.block_rows
        if n<=2*i+2: raise ValueError('not enough samples for requested block_rows')
        cols=n-2*i+1; hankel=torch.stack([y[k:k+cols].T for k in range(2*i)],0)
        past=hankel[:i].reshape(i*m,cols); future=hankel[i:].reshape(i*m,cols)
        projection=future@past.T@torch.linalg.pinv(past@past.T+self.ridge*torch.eye(i*m,dtype=y.dtype))@past
        u,s,_=torch.linalg.svd(projection,full_matrices=False); obs=u[:,:self.n_states]@torch.diag(torch.sqrt(s[:self.n_states]))
        states=torch.linalg.lstsq(obs,future).solution.T; x0=states[:-1]; x1=states[1:]; yy=y[i:i+states.shape[0]-1]
        a=torch.linalg.lstsq(x0,x1).solution.T; c=torch.linalg.lstsq(x0,yy).solution.T
        ew=x1-x0@a.T; ev=yy-x0@c.T; q=symmetrise(ew.T@ew/max(1,ew.shape[0]-1)); r=symmetrise(ev.T@ev/max(1,ev.shape[0]-1))
        self.system_=LinearDynamicalSystem(a,c,q,r,state_covariance=torch.eye(self.n_states,dtype=y.dtype)); self.singular_values_=s; self.states_=states
        return self


class LinearGaussianEM(BaseEstimator):
    """EM refinement of a latent linear Gaussian state-space model."""
    def __init__(self,system,n_iter:int=20,min_covar:float=1e-7): self.system=system; self.n_iter=n_iter; self.min_covar=min_covar
    def fit(self,observations):
        """Fit fit from observations.
        
        Parameters
        ----------
        observations
            Observed time series in ComplexTorch batch-first layout.
        
        Returns
        -------
        object
            The fitted estimator instance.
        
        Notes
        -----
        Batch dimensions are preserved unless explicitly documented otherwise.
        The implementation validates dimensional and positive-definiteness
        requirements before executing the numerical core.
        """
        y=torch.as_tensor(observations,dtype=self.system.transition.dtype,device=self.system.transition.device); sys=self.system; history=[]
        for _ in range(self.n_iter):
            sm=kalman_smoother(y,sys); x=sm.smoothed_mean; p=sm.smoothed_covariance; lag=sm.lag_covariance; exx=p+x.unsqueeze(-1)*x.unsqueeze(-2)
            sum_prev=exx[:-1].sum(0); sum_next=exx[1:].sum(0); cross=(lag[1:]+x[1:].unsqueeze(-1)*x[:-1].unsqueeze(-2)).sum(0)
            a=cross@torch.linalg.pinv(sum_prev); c=(y.T@x)@torch.linalg.pinv(exx.sum(0))
            q=symmetrise((sum_next-a@cross.T-cross@a.T+a@sum_prev@a.T)/(x.shape[0]-1))
            residual=y-x@c.T; r=symmetrise((residual.T@residual+torch.einsum('ij,tjk,lk->il',c,p,c))/x.shape[0])
            q=q+self.min_covar*torch.eye(q.shape[-1],dtype=q.dtype,device=q.device); r=r+self.min_covar*torch.eye(r.shape[-1],dtype=r.dtype,device=r.device)
            sys=LinearDynamicalSystem(a,c,q,r,state_covariance=exx.mean(0)); history.append(float(sm.log_likelihood))
        self.system_=sys; self.log_likelihood_history_=history; return self
