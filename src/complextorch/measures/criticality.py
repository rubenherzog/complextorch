"""Criticality-oriented diagnostics from fitted linear dynamics.

Notes
-----
Stability diagnostics are derived from the companion eigenvalues. The dominant
timescale associated with spectral radius :math:`
ho<1` is

.. math::

   	au=-1/\log
ho.

References
----------
- Lütkepohl, H. (2005), stability of VAR processes.
"""
from __future__ import annotations
import torch
from ..representations import VARSystem

def stability_margin(system:VARSystem)->torch.Tensor:
    """Stability margin.
    
    Parameters
    ----------
    system
        Input controlling ``stability_margin``.
    
    Returns
    -------
    object
        Result described by the function name and annotated return type.
    
    Notes
    -----
    Tensor batch dimensions are preserved unless the public API explicitly
    documents a squeeze operation. Numerical validation is performed by the
    module before the core calculation.
    """
    return 1.0-system.spectral_radius

def dominant_timescale(system:VARSystem,*,sampling_interval:float=1.0)->torch.Tensor:
    """Dominant timescale.
    
    Parameters
    ----------
    system
        Input controlling ``dominant_timescale``.
    sampling_interval
        Input controlling ``dominant_timescale``.
    
    Returns
    -------
    object
        Result described by the function name and annotated return type.
    
    Notes
    -----
    Tensor batch dimensions are preserved unless the public API explicitly
    documents a squeeze operation. Numerical validation is performed by the
    module before the core calculation.
    """
    rho=system.spectral_radius; tiny=torch.finfo(rho.dtype).tiny
    safe=rho.clamp(min=tiny,max=1.0-torch.finfo(rho.dtype).eps); tau=-float(sampling_interval)/torch.log(safe)
    return torch.where(rho<=tiny,torch.zeros_like(tau),tau)

def covariance_amplification(system:VARSystem)->torch.Tensor:
    """Covariance amplification.
    
    Parameters
    ----------
    system
        Input controlling ``covariance_amplification``.
    
    Returns
    -------
    object
        Result described by the function name and annotated return type.
    
    Notes
    -----
    Tensor batch dimensions are preserved unless the public API explicitly
    documents a squeeze operation. Numerical validation is performed by the
    module before the core calculation.
    """
    present=torch.diagonal(system.present_covariance,dim1=-2,dim2=-1).sum(-1)
    innovation=torch.diagonal(system.innovation_covariance,dim1=-2,dim2=-1).sum(-1)
    return present/innovation
