"""Batched Gaussian information-theoretic primitives."""
from __future__ import annotations
import math
import torch
from ..linalg import spd_logdet

def total_correlation(covariance:torch.Tensor,*,base:float=2.0)->torch.Tensor:
    cov=torch.as_tensor(covariance); diagonal=torch.diagonal(cov,dim1=-2,dim2=-1)
    if bool((diagonal<=0).any()): raise ValueError('covariance diagonal must be positive')
    return .5*(torch.log(diagonal).sum(-1)-spd_logdet(cov))/math.log(base)

def gaussian_mutual_information(joint_covariance:torch.Tensor,n_left:int,*,base:float=2.0)->torch.Tensor:
    joint=torch.as_tensor(joint_covariance); n_total=joint.shape[-1]
    if not 0<n_left<n_total: raise ValueError('n_left must split covariance')
    left=joint[...,:n_left,:n_left]; right=joint[...,n_left:,n_left:]
    return .5*(spd_logdet(left)+spd_logdet(right)-spd_logdet(joint))/math.log(base)
