"""Deterministic and random stable VAR systems for validation."""
from __future__ import annotations
import math
import torch
from .linalg import spectral_radius
from .representations import companion_matrix

def _normalise_coefficients(coefficients):
    coef=torch.as_tensor(coefficients)
    if coef.ndim==3: coef=coef.unsqueeze(0)
    if coef.ndim!=4: raise ValueError('coefficients must have shape (p,n,n) or (batch,p,n,n)')
    return coef

def _normalise_covariance(covariance,batch,n):
    q=torch.as_tensor(covariance)
    if q.ndim==2: q=q.unsqueeze(0)
    if q.shape[0]==1 and batch>1: q=q.expand(batch,-1,-1)
    if q.shape!=(batch,n,n): raise ValueError('covariance has incompatible shape')
    return q

def simulate_var(coefficients,innovation_covariance,n_times:int,*,burnin:int=500,seed:int=0):
    coef=_normalise_coefficients(coefficients); batch,order,n,_=coef.shape; q=_normalise_covariance(innovation_covariance,batch,n).to(coef)
    if bool((spectral_radius(companion_matrix(coef))>=1).any()): raise ValueError('simulation requires stable coefficients')
    gen=torch.Generator(device=coef.device); gen.manual_seed(seed); chol=torch.linalg.cholesky(q); total=n_times+burnin
    innovations=torch.randn((batch,total,n),dtype=coef.dtype,device=coef.device,generator=gen)
    innovations=torch.einsum('bij,btj->bti',chol,innovations); x=torch.zeros_like(innovations)
    for t in range(total):
        value=innovations[:,t]
        for lag in range(1,order+1):
            if t>=lag: value=value+torch.einsum('bij,bj->bi',coef[:,lag-1],x[:,t-lag])
        x[:,t]=value
    return x[:,burnin:]

def random_stable_var(batch:int,n_variables:int,order:int,*,spectral_radius_target:float=.85,noise_scale:float=1.,seed:int=0,dtype=torch.float64,device='cpu'):
    if not 0<spectral_radius_target<1: raise ValueError('spectral_radius_target must lie in (0,1)')
    gen=torch.Generator(device=torch.device(device)); gen.manual_seed(seed)
    raw=torch.randn((batch,order,n_variables,n_variables),generator=gen,dtype=dtype,device=device)/math.sqrt(n_variables*order)
    lo=torch.zeros(batch,dtype=dtype,device=device); hi=torch.ones(batch,dtype=dtype,device=device); raw_rho=spectral_radius(companion_matrix(raw)); lo=torch.where(raw_rho<=spectral_radius_target,hi,lo)
    for _ in range(48):
        mid=.5*(lo+hi); rho=spectral_radius(companion_matrix(raw*mid[:,None,None,None])); ok=rho<=spectral_radius_target; lo=torch.where(ok,mid,lo); hi=torch.where(ok,hi,mid)
    coef=raw*lo[:,None,None,None]; q=noise_scale*torch.eye(n_variables,dtype=dtype,device=device).expand(batch,-1,-1).clone(); return coef,q

def _cycle(n,frustrated,*,dtype,device):
    m=torch.zeros((n,n),dtype=dtype,device=device)
    for row in range(n): m[row,(row+1)%n]=1
    if frustrated: m[0,1]=-1
    return m

def demo_var(n_variables:int=3,order:int=2,*,temporal_path:float=-.95,temporal_gain:float=.9,noise_correlation:float=-.25,lag_weights=None,stability_target:float=.98,dtype=torch.float64,device='cpu'):
    dev=torch.device(device); weights=torch.ones(order,dtype=dtype,device=dev) if lag_weights is None else torch.as_tensor(lag_weights,dtype=dtype,device=dev)
    pattern=_cycle(n_variables,temporal_path<0,dtype=dtype,device=dev); eye=torch.eye(n_variables,dtype=dtype,device=dev); w=abs(float(temporal_path)); mats=[]
    for lag in range(order):
        raw=(1-w)*eye+w*torch.linalg.matrix_power(pattern,lag+1); mats.append(weights[lag]*temporal_gain*raw/spectral_radius(raw))
    coef=torch.stack(mats).unsqueeze(0)
    if float(spectral_radius(companion_matrix(coef)))>=stability_target:
        lo,hi=0.,1.
        for _ in range(60):
            mid=.5*(lo+hi)
            if float(spectral_radius(companion_matrix(mid*coef)))<stability_target: lo=mid
            else: hi=mid
        coef=lo*coef
    lower=-1/(n_variables-1)+1e-4; corr=min(.999,max(lower,float(noise_correlation))); q=(1-corr)*eye+corr*torch.ones_like(eye)
    return coef,q.unsqueeze(0)
