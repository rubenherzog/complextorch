"""Gaussian emergence measures for linear macro projections."""
from __future__ import annotations
import math
import torch
from ..linalg import spd_logdet, spd_solve, symmetrise
from ..representations import VARSystem


def _projection(macro_projection:torch.Tensor,system:VARSystem)->torch.Tensor:
    m=torch.as_tensor(macro_projection,dtype=system.coefficients.dtype,device=system.coefficients.device)
    if m.ndim==2: m=m.unsqueeze(0)
    if m.shape[0]==1 and system.batch_size>1: m=m.expand(system.batch_size,-1,-1)
    if m.shape[-1]!=system.n_variables: raise ValueError('macro projection must map micro variables to macro variables')
    return m


def _mi_from_cov(a:torch.Tensor,b:torch.Tensor,cross:torch.Tensor)->torch.Tensor:
    cond=symmetrise(a-cross@spd_solve(b,cross.transpose(-1,-2)))
    return 0.5*(spd_logdet(a)-spd_logdet(cond))/math.log(2.0)


def emergence_measures(system:VARSystem,macro_projection:torch.Tensor)->dict[str,torch.Tensor]:
    """Return Ψ, Δ and Γ for Y=M X.

    Ψ = I(Y_t; X_past) - I(Y_t; Y_past)
    Δ = I(Y_t; Y_past) - sum_j I(Y_t^j; Y_past^j)
    Γ = I(Y_t; X_past) - sum_j I(Y_t^j; X_past)
    """
    m=_projection(macro_projection,system); p=system.projection; s=system.state_covariance
    current=m@p
    sy=current@s@current.transpose(-1,-2); cross_full=current@system.companion@s
    i_full=_mi_from_cov(sy,s,cross_full)
    b,m_dim,_=m.shape; n=system.n_variables
    mp=torch.zeros((b,m_dim*system.order,n*system.order),dtype=m.dtype,device=m.device)
    for k in range(system.order): mp[:,k*m_dim:(k+1)*m_dim,k*n:(k+1)*n]=m
    syp=mp@s@mp.transpose(-1,-2); cross_macro=current@system.companion@s@mp.transpose(-1,-2)
    i_macro=_mi_from_cov(sy,syp,cross_macro)
    self_terms=[]; full_parts=[]
    for j in range(m_dim):
        yj=sy[...,j:j+1,j:j+1]
        row=cross_macro[...,j:j+1,:]
        own_idx=torch.tensor([j+k*m_dim for k in range(system.order)],device=s.device)
        own=syp.index_select(-2,own_idx).index_select(-1,own_idx); own_cross=row.index_select(-1,own_idx)
        self_terms.append(_mi_from_cov(yj,own,own_cross))
        full_parts.append(_mi_from_cov(yj,s,cross_full[...,j:j+1,:]))
    delta=i_macro-torch.stack(self_terms,-1).sum(-1)
    gamma=i_full-torch.stack(full_parts,-1).sum(-1)
    return {'psi':i_full-i_macro,'delta':delta,'gamma':gamma,'macro_predictive_information':i_macro,'macro_from_micro_predictive_information':i_full}


def emergence_from_observations(current:torch.Tensor,past:torch.Tensor,macro_projection:torch.Tensor)->dict[str,torch.Tensor]:
    """Estimate Ψ from observations using Gaussian covariance plug-in estimates."""
    x=torch.as_tensor(current); z=torch.as_tensor(past,dtype=x.dtype,device=x.device); m=torch.as_tensor(macro_projection,dtype=x.dtype,device=x.device)
    if x.ndim!=2 or z.ndim!=2: raise ValueError('current and past must be 2-D observations')
    y=x@m.T
    def cov(v):
        v=v-v.mean(0); return v.T@v/(v.shape[0]-1)
    joint=torch.cat([y,z],-1); sj=cov(joint); sy=cov(y); sz=cov(z); cross=sj[:y.shape[1],y.shape[1]:]
    i_full=_mi_from_cov(sy,sz,cross)
    if z.shape[1]%x.shape[1]!=0: raise ValueError('past dimension must be a multiple of micro dimension')
    order=z.shape[1]//x.shape[1]; mp=torch.block_diag(*([m]*order)); yp=z@mp.T
    sjm=cov(torch.cat([y,yp],-1)); syp=cov(yp); cm=sjm[:y.shape[1],y.shape[1]:]; i_macro=_mi_from_cov(sy,syp,cm)
    return {'psi':i_full-i_macro,'macro_predictive_information':i_macro,'macro_from_micro_predictive_information':i_full}
