r"""Inference and constrained-reference procedures for model-derived measures.

Sampling uncertainty and null-reference normalization are intentionally kept as
separate scientific operations. :func:`measure_confidence_intervals` provides
ARres-inspired bootstrap/parametric confidence intervals, while
:func:`numit_pid_var` compares Gaussian VAR PID atoms with a TMI-matched NuMIT
null ensemble. NuMIT is not a resampling method and is therefore not part of
:class:`ResamplingMethod`.

The private bootstrap/refit mechanics live in :mod:`complextorch._resampling`;
measure registration lives in :mod:`complextorch.inference_registry`. Dynamical
dependence is evaluated only for a supplied fixed projection or fixed batch of
projections; projection optimization is never performed inside resampling.

References
----------
- Beda, A., Simpson, D. M., and Faes, L. (2017). Estimation of confidence
  limits for descriptive indexes derived from autoregressive analysis of time
  series. *PLoS ONE*, 12(10), e0186694.
- Liardi, A. et al. (2025). Null models for comparing information decomposition
  across complex systems. *PLoS Computational Biology*, 21, e1013629.
"""
from ._resampling import ConfidenceIntervalResult, MeasureInterval, ResamplingMethod
from .confidence import measure_confidence_intervals
from .inference_registry import InferenceMeasureConfig
from .numit import NuMITPIDResult, numit_pid_var

__all__ = [
    "ConfidenceIntervalResult",
    "InferenceMeasureConfig",
    "MeasureInterval",
    "NuMITPIDResult",
    "ResamplingMethod",
    "measure_confidence_intervals",
    "numit_pid_var",
]
