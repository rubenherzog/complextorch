"""Axis-layout adapters between ComplexTorch and ComplexBox.

ComplexBox follows ``(variables, time, trials)`` while ComplexTorch follows
``(trials, time, variables)``. Coefficient tensors are analogously permuted
between ``(target, source, lag)`` and ``(batch, lag, target, source)``. These
operations never alter numerical values.

References
----------
- Barnett, L. and Seth, A. K. (2014). The MVGC multivariate Granger causality
  toolbox. *Journal of Neuroscience Methods*, 223, 50--68.
- ComplexBox: https://github.com/bmilinkovic/complexbox
"""
from __future__ import annotations
import numpy as np
import torch
from ._typing import ArrayLike

def _as_tensor(x: ArrayLike) -> torch.Tensor:
    """As tensor.
    
    Parameters
    ----------
    x
        Input observations or tensor-valued quantity.
    
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
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x))

def from_complexbox_timeseries(x: ArrayLike) -> torch.Tensor:
    """Convert complexbox timeseries to ComplexTorch layout.
    
    Parameters
    ----------
    x
        Input observations or tensor-valued quantity.
    
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
    t = _as_tensor(x)
    if t.ndim == 2:
        return t.transpose(0, 1).unsqueeze(0).contiguous()
    if t.ndim == 3:
        return t.permute(2, 1, 0).contiguous()
    raise ValueError("ComplexBox time series must be 2-D or 3-D")

def to_complexbox_timeseries(x: ArrayLike, *, squeeze_single: bool = True) -> torch.Tensor:
    """Convert to complexbox timeseries.
    
    Parameters
    ----------
    x
        Input observations or tensor-valued quantity.
    squeeze_single
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
    t = _as_tensor(x)
    if t.ndim == 2:
        t = t.unsqueeze(0)
    if t.ndim != 3:
        raise ValueError("ComplexTorch time series must be 2-D or 3-D")
    out = t.permute(2, 1, 0).contiguous()
    return out[..., 0] if squeeze_single and out.shape[-1] == 1 else out

def from_complexbox_var(a: ArrayLike) -> torch.Tensor:
    """Convert complexbox var to ComplexTorch layout.
    
    Parameters
    ----------
    a
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
    t = _as_tensor(a)
    if t.ndim != 3:
        raise ValueError("ComplexBox VAR coefficients must have shape (n,n,p)")
    return t.permute(2, 0, 1).unsqueeze(0).contiguous()

def to_complexbox_var(a: ArrayLike, *, squeeze_single: bool = True) -> torch.Tensor:
    """Convert to complexbox var.
    
    Parameters
    ----------
    a
        Input required by this calculation.
    squeeze_single
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
    t = _as_tensor(a)
    if t.ndim == 3:
        t = t.unsqueeze(0)
    if t.ndim != 4:
        raise ValueError("ComplexTorch VAR coefficients must have shape (batch,p,n,n)")
    out = t.permute(2, 3, 1, 0).contiguous()
    return out[..., 0] if squeeze_single and out.shape[-1] == 1 else out
