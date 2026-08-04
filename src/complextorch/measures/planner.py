"""Dependency-aware measure planner with shared intermediate caching."""
from __future__ import annotations
from collections.abc import Iterable
import torch
from ..representations import VARSystem
from .cmem import compute_cmem
from .criticality import covariance_amplification, dominant_timescale, stability_margin

class DynamicalMeasures:
    AVAILABLE={"spectral_radius","stability_margin","dominant_timescale","covariance_amplification","stationary_covariance","cmem3_total","cmem1_total","cmem3_lag","cmem3_curve","cmem1_curve","tc_innovation","tc_present"}
    def __init__(self,measures:Iterable[str],*,tau_max:int=10):
        self.measures=tuple(measures); self.tau_max=tau_max; unknown=set(self.measures)-self.AVAILABLE
        if unknown: raise ValueError(f"unknown measures: {sorted(unknown)}")
    def __call__(self,system:VARSystem)->dict[str,torch.Tensor]:
        result={}; cmem_names={"cmem3_total","cmem1_total","cmem3_lag","cmem3_curve","cmem1_curve","tc_innovation","tc_present"}; cmem=compute_cmem(system,self.tau_max) if cmem_names.intersection(self.measures) else None
        for name in self.measures:
            if name=="spectral_radius": value=system.spectral_radius
            elif name=="stability_margin": value=stability_margin(system)
            elif name=="dominant_timescale": value=dominant_timescale(system)
            elif name=="covariance_amplification": value=covariance_amplification(system)
            elif name=="stationary_covariance": value=system.present_covariance
            else: value=getattr(cmem,name)
            result[name]=value
        return result
