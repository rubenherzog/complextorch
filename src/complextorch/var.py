"""Batched closed-form VAR estimation with a scikit-learn compatible surface."""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
from typing import Literal
import numpy as np
import torch
from sklearn.base import BaseEstimator
from ._typing import ArrayLike
from .linalg import stable_cholesky
from .representations import LinearDynamicalSystem, VARSystem, build_var_system

@dataclass(frozen=True)
class VARParameters:
    coefficients: torch.Tensor
    intercept: torch.Tensor
    innovation_covariance: torch.Tensor
    residuals: torch.Tensor
    n_observations: int
    order: int
    fit_mode: str
    solver: str
    fit_time: float

class VAR(BaseEstimator):
    def __init__(self, order:int=1, *, alpha:float=0.0, fit_intercept:bool=True, mode:Literal['independent','pooled']='independent', solver:Literal['auto','lstsq','cholesky','pinv']='auto', covariance:Literal['unbiased','mle']='unbiased', device:str='auto', dtype:str='float64', stability:Literal['check','ignore']='check'):
        self.order=order; self.alpha=alpha; self.fit_intercept=fit_intercept; self.mode=mode; self.solver=solver; self.covariance=covariance; self.device=device; self.dtype=dtype; self.stability=stability
    @staticmethod
    def _resolve_dtype(name:str)->torch.dtype:
        dtype=getattr(torch,name,None)
        if dtype not in (torch.float32,torch.float64): raise ValueError("dtype must be 'float32' or 'float64'")
        return dtype
    def _resolve_device(self)->torch.device:
        if self.device=='auto': return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dev=torch.device(self.device)
        if dev.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA was requested but is not available')
        return dev
    def _normalise_input(self,x:ArrayLike)->torch.Tensor:
        t=torch.as_tensor(x,dtype=self._resolve_dtype(self.dtype),device=self._resolve_device())
        if t.ndim==2: t=t.unsqueeze(0)
        if t.ndim!=3: raise ValueError('X must have shape (time,variables) or (batch,time,variables)')
        if not bool(torch.isfinite(t).all()): raise ValueError('X contains NaN or infinite values')
        if t.shape[1]<=self.order: raise ValueError('time dimension must exceed VAR order')
        return t.contiguous()
    @staticmethod
    def lagged_design(x:torch.Tensor,order:int):
        batch,time,_=x.shape; y=x[:,order:,:]
        blocks=[x[:,order-lag:time-lag,:] for lag in range(1,order+1)]
        return torch.cat(blocks,dim=-1),y
    def _choose_solver(self): return self.solver if self.solver!='auto' else ('lstsq' if self.alpha==0 else 'cholesky')
    def _solve_cholesky(self,design,targets):
        gram=design.transpose(-1,-2)@design; rhs=design.transpose(-1,-2)@targets
        penalty=torch.eye(gram.shape[-1],dtype=gram.dtype,device=gram.device)
        if self.fit_intercept: penalty[-1,-1]=0
        chol,_=stable_cholesky(gram+float(self.alpha)*penalty,jitter=1e-12)
        return torch.cholesky_solve(rhs,chol)
    def _fit_tensor(self,x):
        if self.order<1 or self.alpha<0: raise ValueError('invalid order or alpha')
        design,targets=self.lagged_design(x,self.order); batch,nobs,n=targets.shape
        if self.fit_intercept: design=torch.cat([design,torch.ones((batch,nobs,1),dtype=x.dtype,device=x.device)],-1)
        if self.mode=='pooled': design_fit=design.reshape(1,batch*nobs,-1); targets_fit=targets.reshape(1,batch*nobs,n)
        elif self.mode=='independent': design_fit=design; targets_fit=targets
        else: raise ValueError("mode must be 'independent' or 'pooled'")
        solver=self._choose_solver()
        if solver=='lstsq': solution=torch.linalg.lstsq(design_fit,targets_fit).solution
        elif solver=='pinv': solution=torch.linalg.pinv(design_fit)@targets_fit
        elif solver=='cholesky': solution=self._solve_cholesky(design_fit,targets_fit)
        else: raise ValueError('unknown solver')
        if self.fit_intercept: coef_flat=solution[:,:-1,:]; intercept=solution[:,-1,:]
        else: coef_flat=solution; intercept=torch.zeros((solution.shape[0],n),dtype=x.dtype,device=x.device)
        coefficients=coef_flat.reshape(solution.shape[0],self.order,n,n).transpose(-1,-2)
        residuals=targets_fit-design_fit@solution; nfit=targets_fit.shape[1]; predictors=design_fit.shape[-1]
        denominator=nfit if self.covariance=='mle' else nfit-predictors
        if denominator<=0: raise ValueError('not enough observations')
        cov=residuals.transpose(-1,-2)@residuals/float(denominator)
        return VARParameters(coefficients,intercept,cov,residuals,nfit,self.order,self.mode,solver,0.0)
    def fit(self,X:ArrayLike,y=None):
        del y; x=self._normalise_input(X); start=perf_counter(); p=self._fit_tensor(x); elapsed=perf_counter()-start
        self.params_=VARParameters(p.coefficients,p.intercept,p.innovation_covariance,p.residuals,p.n_observations,p.order,p.fit_mode,p.solver,elapsed)
        self.coef_=p.coefficients; self.intercept_=p.intercept; self.noise_covariance_=p.innovation_covariance; self.residuals_=p.residuals; self.n_features_in_=x.shape[-1]; self.n_epochs_in_=x.shape[0]; self.device_=x.device; self.dtype_=x.dtype; self.fit_time_=elapsed
        if self.stability=='check':
            try: system=build_var_system(self.coef_,self.noise_covariance_); self.spectral_radius_=system.spectral_radius; self.is_stable_=self.spectral_radius_<1
            except ValueError:
                from .representations import companion_matrix
                from .linalg import spectral_radius
                self.spectral_radius_=spectral_radius(companion_matrix(self.coef_)); self.is_stable_=self.spectral_radius_<1
        return self
    def _check_fitted(self):
        if not hasattr(self,'params_'): raise RuntimeError('estimator is not fitted')
    def one_step_predictions(self,X):
        self._check_fitted(); x=self._normalise_input(X); design,_=self.lagged_design(x,self.order); batch,nobs,_=design.shape
        if self.fit_intercept: design=torch.cat([design,torch.ones((batch,nobs,1),dtype=x.dtype,device=x.device)],-1)
        coef_flat=self.coef_.transpose(-1,-2).reshape(self.coef_.shape[0],-1,x.shape[-1]); solution=torch.cat([coef_flat,self.intercept_[:,None,:]],1) if self.fit_intercept else coef_flat
        if self.mode=='pooled': solution=solution.expand(batch,-1,-1)
        elif solution.shape[0]!=batch: raise ValueError('number of epochs differs')
        return design@solution
    def predict(self,X): return self.one_step_predictions(X).detach().cpu().numpy()
    def forecast(self,history,steps:int):
        self._check_fitted(); hist=self._normalise_input(history)
        if steps<1 or hist.shape[1]<self.order: raise ValueError('invalid forecast request')
        batch=hist.shape[0]; coef=self.coef_; intercept=self.intercept_
        if self.mode=='pooled': coef=coef.expand(batch,-1,-1,-1); intercept=intercept.expand(batch,-1)
        state=hist[:,-self.order:,:].clone(); out=[]
        for _ in range(steps):
            nxt=intercept.clone()
            for lag in range(self.order): nxt=nxt+torch.einsum('bij,bj->bi',coef[:,lag],state[:,-(lag+1)])
            out.append(nxt); state=torch.cat([state[:,1:],nxt[:,None,:]],1)
        return torch.stack(out,1)
    def residuals(self,X):
        x=self._normalise_input(X); _,targets=self.lagged_design(x,self.order); return targets-self.one_step_predictions(x)
    def gaussian_nll(self,X,*,reduction='mean'):
        errors=self.residuals(X); cov=self.noise_covariance_
        if self.mode=='pooled': cov=cov.expand(errors.shape[0],-1,-1)
        chol,_=stable_cholesky(cov,jitter=1e-10); solved=torch.cholesky_solve(errors.unsqueeze(-1),chol[:,None,:,:]).squeeze(-1)
        values=.5*((errors*solved).sum(-1)+2*torch.log(torch.diagonal(chol,dim1=-2,dim2=-1)).sum(-1)[:,None]+errors.shape[-1]*np.log(2*np.pi))
        if reduction=='none': return values
        if reduction=='mean': return values.mean()
        if reduction=='sum': return values.sum()
        raise ValueError('bad reduction')
    def score(self,X,y=None): del y; return -float(self.gaussian_nll(X))
    def to_var_system(self,*,lyapunov_method='doubling')->VARSystem: self._check_fitted(); return build_var_system(self.coef_,self.noise_covariance_,lyapunov_method=lyapunov_method)
    def to_state_space(self,*,lyapunov_method='doubling')->LinearDynamicalSystem: return self.to_var_system(lyapunov_method=lyapunov_method).to_state_space()
