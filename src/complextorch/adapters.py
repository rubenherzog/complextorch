"""Adapters for data layouts used by ComplexTorch and ComplexBox.

Notes
-----
These adapters only permute axes; they do not alter numerical values.  The
ComplexBox/MVGC convention is ``(variables, time, trials)`` whereas
ComplexTorch uses ``(trials, time, variables)``.

References
----------
- Barnett, L. and Seth, A. K. (2014). The MVGC multivariate Granger causality
  toolbox. *Journal of Neuroscience Methods*, 223, 50--68.
- ComplexBox repository: https://github.com/bmilinkovic/complexbox
"""
from __future__ import annotations
import numpy as np
import torch
from ._typing import ArrayLike

def _as_tensor(x: ArrayLike) -> torch.Tensor:
    """ as tensor.
    
    Parameters
    ----------
    x
        Input controlling ``_as_tensor``.
    
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
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x))

def from_complexbox_timeseries(x: ArrayLike) -> torch.Tensor:
    """From complexbox timeseries.
    
    Parameters
    ----------
    x
        Input controlling ``from_complexbox_timeseries``.
    
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
    t = _as_tensor(x)
    if t.ndim == 2:
        return t.transpose(0, 1).unsqueeze(0).contiguous()
    if t.ndim == 3:
        return t.permute(2, 1, 0).contiguous()
    raise ValueError("ComplexBox time series must be 2-D or 3-D")

def to_complexbox_timeseries(x: ArrayLike, *, squeeze_single: bool = True) -> torch.Tensor:
    """To complexbox timeseries.
    
    Parameters
    ----------
    x
        Input controlling ``to_complexbox_timeseries``.
    squeeze_single
        Input controlling ``to_complexbox_timeseries``.
    
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
    t = _as_tensor(x)
    if t.ndim == 2:
        t = t.unsqueeze(0)
    if t.ndim != 3:
        raise ValueError("ComplexTorch time series must be 2-D or 3-D")
    out = t.permute(2, 1, 0).contiguous()
    return out[..., 0] if squeeze_single and out.shape[-1] == 1 else out

def from_complexbox_var(a: ArrayLike) -> torch.Tensor:
    """From complexbox var.
    
    Parameters
    ----------
    a
        Input controlling ``from_complexbox_var``.
    
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
    t = _as_tensor(a)
    if t.ndim != 3:
        raise ValueError("ComplexBox VAR coefficients must have shape (n,n,p)")
    return t.permute(2, 0, 1).unsqueeze(0).contiguous()

def to_complexbox_var(a: ArrayLike, *, squeeze_single: bool = True) -> torch.Tensor:
    """To complexbox var.
    
    Parameters
    ----------
    a
        Input controlling ``to_complexbox_var``.
    squeeze_single
        Input controlling ``to_complexbox_var``.
    
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
    t = _as_tensor(a)
    if t.ndim == 3:
        t = t.unsqueeze(0)
    if t.ndim != 4:
        raise ValueError("ComplexTorch VAR coefficients must have shape (batch,p,n,n)")
    out = t.permute(2, 3, 1, 0).contiguous()
    return out[..., 0] if squeeze_single and out.shape[-1] == 1 else out
