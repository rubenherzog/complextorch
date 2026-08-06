"""Public MVGC dispatch preserving legacy observation-based calls.

The canonical path is model-first. Calls whose first argument is a canonical
VAR/state-space object are routed to :mod:`complextorch.measures.primary`.
Legacy calls beginning with an observations tensor are routed to the explicitly
named secondary estimators.

Notes
-----
This compatibility layer dispatches canonical model inputs to analytical MVGC
and legacy observation inputs to explicitly empirical estimators.

References
----------
- Barnett, L. and Seth, A. K. (2014), MVGC toolbox paper.
"""
from __future__ import annotations

import warnings
import torch

from ..control import InnovationsStateSpace
from ..representations import LinearDynamicalSystem, VARSystem
from .primary import spectral_mvgc as model_spectral_mvgc
from .primary import temporal_mvgc as model_temporal_mvgc
from .secondary import estimate_spectral_mvgc_from_observations
from .secondary import estimate_temporal_mvgc_from_observations

_MODEL_TYPES = (VARSystem, LinearDynamicalSystem, InnovationsStateSpace)


def temporal_mvgc(model_or_observations, *args, **kwargs):
    """Compute model-based MVGC, with deprecated observation-call dispatch.
        
        Compute conditional time-domain multivariate Granger causality.
        
        .. math:: F_{Y	o X\mid Z}=\log(\det\Sigma^R_{XX}/\det\Sigma_{XX}).
        
        References
        ----------
        Geweke (1982); Barnett and Seth (2014, 2015).
    
    Compute conditional time-domain multivariate Granger causality.
    
    .. math:: F_{Y	o X\mid Z}=\log(\det\Sigma^R_{XX}/\det\Sigma_{XX}).
    
    References
    ----------
    Geweke (1982); Barnett and Seth (2014, 2015).
    """
    if isinstance(model_or_observations, _MODEL_TYPES):
        return model_temporal_mvgc(model_or_observations, *args, **kwargs)
    if isinstance(model_or_observations, torch.Tensor):
        warnings.warn(
            "temporal_mvgc(observations, ...) is deprecated; use "
            "estimate_temporal_mvgc_from_observations(...) from "
            "complextorch.measures.secondary",
            DeprecationWarning,
            stacklevel=2,
        )
        return estimate_temporal_mvgc_from_observations(
            model_or_observations, *args, **kwargs
        )
    raise TypeError("first argument must be a canonical model or observations tensor")


def spectral_mvgc(model_or_observations, *args, **kwargs):
    """Compute model-based spectral MVGC, with legacy observation dispatch.
        
        Compute conditional spectral multivariate Granger causality.
        
        The frequency-resolved decomposition is obtained from innovations-form transfer
        functions and integrates to temporal GC.
        
        References
        ----------
        Geweke (1982); Barnett and Seth (2014, 2015).
    
    Compute conditional spectral multivariate Granger causality.
    
    The frequency-resolved decomposition is obtained from innovations-form transfer
    functions and integrates to temporal GC.
    
    References
    ----------
    Geweke (1982); Barnett and Seth (2014, 2015).
    """
    if isinstance(model_or_observations, _MODEL_TYPES):
        return model_spectral_mvgc(model_or_observations, *args, **kwargs)
    if isinstance(model_or_observations, torch.Tensor):
        warnings.warn(
            "spectral_mvgc(observations, ...) is deprecated; use "
            "estimate_spectral_mvgc_from_observations(...) from "
            "complextorch.measures.secondary",
            DeprecationWarning,
            stacklevel=2,
        )
        return estimate_spectral_mvgc_from_observations(
            model_or_observations, *args, **kwargs
        )
    raise TypeError("first argument must be a canonical model or observations tensor")


__all__ = ["temporal_mvgc", "spectral_mvgc"]
