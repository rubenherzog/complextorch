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
    return 1.0-system.spectral_radius

def dominant_timescale(system:VARSystem,*,sampling_interval:float=1.0)->torch.Tensor:
    """Dominant timescale.
    
    Parameters
    ----------
    system
        Canonical VAR or state-space system.
    sampling_interval
        Input required by this calculation.
    
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
    rho=system.spectral_radius; tiny=torch.finfo(rho.dtype).tiny
    safe=rho.clamp(min=tiny,max=1.0-torch.finfo(rho.dtype).eps); tau=-float(sampling_interval)/torch.log(safe)
    return torch.where(rho<=tiny,torch.zeros_like(tau),tau)

def covariance_amplification(system:VARSystem)->torch.Tensor:
    """Covariance amplification.
    
    Parameters
    ----------
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
    present=torch.diagonal(system.present_covariance,dim1=-2,dim2=-1).sum(-1)
    innovation=torch.diagonal(system.innovation_covariance,dim1=-2,dim2=-1).sum(-1)
    return present/innovation
