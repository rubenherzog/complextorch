"""Public dispatch layer for model-based and observation-based MVGC.

Canonical VAR/state-space inputs are routed to analytical primary measures;
finite observations are routed to explicitly empirical secondary estimators.

References
----------
- Barnett, L. and Seth, A. K. (2014, 2015).
"""
from __future__ import annotations

import warnings
import torch

from ..control import InnovationsStateSpace
from ..representations import StateSpaceModel, VARSystem
from .primary import spectral_mvgc as model_spectral_mvgc
from .primary import temporal_mvgc as model_temporal_mvgc
from .secondary import estimate_spectral_mvgc_from_observations
from .secondary import estimate_temporal_mvgc_from_observations

_MODEL_TYPES = (VARSystem, StateSpaceModel, InnovationsStateSpace)


def temporal_mvgc(model_or_observations, *args, **kwargs):
    """Compute conditional time-domain multivariate Granger causality.
    
    .. math::
    
       F_{Y\to X\mid Z}
       =\log\frac{\det\Sigma^{R}_{XX}}{\det\Sigma_{XX}}.
    
    References
    ----------
    - Geweke (1982); Barnett and Seth (2014, 2015).
    """
    # Compare full and reduced innovation covariance volumes to obtain Geweke time-domain Granger causality.
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
    """Compute conditional spectral multivariate Granger causality.
    
    The frequency-resolved decomposition is obtained from innovations-form transfer
    functions and integrates to temporal GC.
    
    References
    ----------
    - Geweke (1982); Barnett and Seth (2014, 2015).
    """
    # Decompose the predictive covariance ratio over frequency using the model transfer function and spectrum.
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
