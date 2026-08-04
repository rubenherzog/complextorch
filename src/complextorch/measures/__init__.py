"""Complexity measures organised into model-first and empirical tiers.

Use :mod:`complextorch.measures.primary` for the main analytical API.  The
legacy direct imports remain available for backward compatibility, while
sample-based estimators are grouped explicitly in
:mod:`complextorch.measures.secondary`.
"""
from . import primary, secondary
from .primary import (
    Model,
    ModelMeasureConfig,
    compute_all_model_measures,
    gaussian_measures_from_model,
    pairwise_gaussian_mutual_information,
    past_future_covariance,
    phiid_from_model,
    stationary_observation_covariance,
)
from .gaussian import gaussian_entropy, conditional_covariance, gaussian_mutual_information, gaussian_conditional_mutual_information, total_correlation, dual_total_correlation, o_information, s_information, local_gaussian_mutual_information
from .dynamics import autocovariances, entropy_rate, predictive_information, active_information_storage, transfer_function, inverse_transfer_function, cross_spectral_density, spectral_entropy
from .emergence import emergence_measures, emergence_from_observations
from .cmem import CMemResult, cmem1_curve, cmem1_total, cmem3_curve, cmem3_lag_decomposition, cmem3_total, compute_cmem
from .criticality import covariance_amplification, dominant_timescale, stability_margin
from .mvgc import temporal_mvgc, spectral_mvgc, state_space_temporal_mvgc, state_space_spectral_mvgc, integrate_spectral_mvgc, pairwise_spectral_gc
from .discrete import discrete_entropy, discrete_mutual_information, discrete_total_correlation, lempel_ziv_complexity
from .phid import gaussian_phiid_mmi, gaussian_phiid_atoms
from .planner import DynamicalMeasures
from .registry import MEASURE_REGISTRY, PRIMARY_MEASURES, SECONDARY_MEASURES
__all__ = [name for name in globals() if not name.startswith("_")]
