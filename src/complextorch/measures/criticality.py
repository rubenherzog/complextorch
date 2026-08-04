"""Criticality-oriented diagnostics from fitted linear dynamics."""
from __future__ import annotations
import torch
from ..representations import VARSystem

def stability_margin(system:VARSystem)->torch.Tensor:
    return 1.0-system.spectral_radius

def dominant_timescale(system:VARSystem,*,sampling_interval:float=1.0)->torch.Tensor:
    rho=system.spectral_radius; tiny=torch.finfo(rho.dtype).tiny
    safe=rho.clamp(min=tiny,max=1.0-torch.finfo(rho.dtype).eps); tau=-float(sampling_interval)/torch.log(safe)
    return torch.where(rho<=tiny,torch.zeros_like(tau),tau)

def covariance_amplification(system:VARSystem)->torch.Tensor:
    present=torch.diagonal(system.present_covariance,dim1=-2,dim2=-1).sum(-1)
    innovation=torch.diagonal(system.innovation_covariance,dim1=-2,dim2=-1).sum(-1)
    return present/innovation
