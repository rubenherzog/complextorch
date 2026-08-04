"""Analytical CMem quantities for stationary Gaussian VAR systems."""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from ..linalg import spd_logdet, spd_solve, symmetrise
from ..representations import VARSystem
from .gaussian import gaussian_mutual_information, total_correlation

@dataclass(frozen=True)
class CMemResult:
    cmem3_total: torch.Tensor
    cmem1_total: torch.Tensor
    cmem3_lag: torch.Tensor
    cmem3_curve: torch.Tensor
    cmem1_curve: torch.Tensor
    tc_innovation: torch.Tensor
    tc_present: torch.Tensor

def cmem3_total(system:VARSystem)->torch.Tensor:
    return total_correlation(system.innovation_covariance)-total_correlation(system.present_covariance)

def cmem1_total(system:VARSystem)->torch.Tensor:
    sigma=system.present_covariance; q=system.innovation_covariance
    full=.5*(spd_logdet(sigma)-spd_logdet(q))/math.log(2)
    parts=.5*torch.log2(torch.diagonal(sigma,dim1=-2,dim2=-1)/torch.diagonal(q,dim1=-2,dim2=-1)).sum(-1)
    return full-parts

def _gamma(system,power): return system.projection@power@system.state_covariance@system.projection.transpose(-1,-2)
def _joint_from_gamma(sigma,gamma): return torch.cat([torch.cat([sigma,gamma],-1),torch.cat([gamma.transpose(-1,-2),sigma],-1)],-2)

def cmem3_curve(system:VARSystem,tau_max:int)->torch.Tensor:
    if tau_max<1: raise ValueError('tau_max must be >=1')
    sigma=system.present_covariance; tc=total_correlation(sigma); power=torch.eye(system.companion.shape[-1],dtype=system.companion.dtype,device=system.companion.device).expand(system.batch_size,-1,-1); values=[]
    for _ in range(tau_max):
        power=power@system.companion; gamma=_gamma(system,power); conditional=symmetrise(sigma-gamma@spd_solve(sigma,gamma.transpose(-1,-2))); values.append(total_correlation(conditional)-tc)
    return torch.stack(values,-1)

def _self_mi(joint,n):
    a=torch.diagonal(joint[...,:n,:n],dim1=-2,dim2=-1); b=torch.diagonal(joint[...,n:,n:],dim1=-2,dim2=-1); c=torch.diagonal(joint[...,:n,n:],dim1=-2,dim2=-1)
    det=(a*b-c.square()).clamp_min(torch.finfo(joint.dtype).tiny); return .5*torch.log2(a*b/det)

def cmem1_curve(system:VARSystem,tau_max:int)->torch.Tensor:
    n=system.n_variables; power=torch.eye(system.companion.shape[-1],dtype=system.companion.dtype,device=system.companion.device).expand(system.batch_size,-1,-1); values=[]
    for _ in range(tau_max):
        power=power@system.companion; joint=_joint_from_gamma(system.present_covariance,_gamma(system,power)); values.append(gaussian_mutual_information(joint,n)-_self_mi(joint,n).sum(-1))
    return torch.stack(values,-1)

def _lagged_covariances(system,max_lag):
    power=torch.eye(system.companion.shape[-1],dtype=system.companion.dtype,device=system.companion.device).expand(system.batch_size,-1,-1); gammas=[system.present_covariance]
    for _ in range(max_lag): power=power@system.companion; gammas.append(_gamma(system,power))
    return gammas

def _joint_cov_lags(system,tau,gammas):
    blocks=[]
    for left in range(tau+1):
        row=[]
        for right in range(tau+1):
            block=gammas[abs(right-left)]; row.append(block.transpose(-1,-2) if right<left else block)
        blocks.append(torch.cat(row,-1))
    return torch.cat(blocks,-2)

def _node_vs_vector_mi(joint,n):
    past=joint[...,n:,n:]; ld=spd_logdet(past); vals=[]
    for node in range(n):
        idx=torch.tensor([node,*range(n,2*n)],device=joint.device); sub=joint.index_select(-2,idx).index_select(-1,idx)
        vals.append(.5*(torch.log(joint[...,node,node])+ld-spd_logdet(sub))/math.log(2))
    return torch.stack(vals,-1)

def _cmem3_lag_one(system,tau,gammas):
    n=system.n_variables; full=_joint_cov_lags(system,tau,gammas); present=torch.arange(0,n,device=full.device); target=torch.arange(tau*n,(tau+1)*n,device=full.device); joint_idx=torch.cat([present,target])
    if tau==1: conditional=full.index_select(-2,joint_idx).index_select(-1,joint_idx)
    else:
        condition=torch.arange(n,tau*n,device=full.device); ordered=torch.cat([joint_idx,condition]); reordered=full.index_select(-2,ordered).index_select(-1,ordered); nj=2*n; s11=reordered[...,:nj,:nj]; s12=reordered[...,:nj,nj:]; s22=reordered[...,nj:,nj:]; conditional=symmetrise(s11-s12@spd_solve(s22,s12.transpose(-1,-2)))
    return gaussian_mutual_information(conditional,n)-_node_vs_vector_mi(conditional,n).sum(-1)

def cmem3_lag_decomposition(system:VARSystem)->torch.Tensor:
    gammas=_lagged_covariances(system,system.order); return torch.stack([_cmem3_lag_one(system,t,gammas) for t in range(1,system.order+1)],-1)

def compute_cmem(system:VARSystem,tau_max:int)->CMemResult:
    return CMemResult(cmem3_total(system),cmem1_total(system),cmem3_lag_decomposition(system),cmem3_curve(system,tau_max),cmem1_curve(system,tau_max),total_correlation(system.innovation_covariance),total_correlation(system.present_covariance))
