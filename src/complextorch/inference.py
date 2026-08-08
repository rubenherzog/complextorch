r"""Resampling-based confidence intervals for model-derived VAR measures.

This public module exposes the ARres-inspired confidence-interval API and result
containers. The private bootstrap/refit mechanics live in
:mod:`complextorch._resampling`; measure registration lives in
:mod:`complextorch.inference_registry`. This separation guarantees that a single
shared VAR ensemble is reused across all requested compatible measures.

Dynamical dependence is evaluated only for a supplied fixed projection or fixed
batch of projections. Projection optimization is never performed inside the
resampling procedure.

References
----------
- Beda, A., Simpson, D. M., and Faes, L. (2017). Estimation of confidence
  limits for descriptive indexes derived from autoregressive analysis of time
  series. *PLoS ONE*, 12(10), e0186694.
"""
from ._resampling import ConfidenceIntervalResult, MeasureInterval, ResamplingMethod
from .confidence import measure_confidence_intervals
from .inference_registry import InferenceMeasureConfig

__all__ = [
    "ConfidenceIntervalResult",
    "InferenceMeasureConfig",
    "MeasureInterval",
    "ResamplingMethod",
    "measure_confidence_intervals",
]
