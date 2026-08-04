"""Secondary empirical measures and sample-based estimation helpers.

These functions operate directly on observations, discretised sequences or
sample covariances. They are intentionally separated from the model-first
primary API because they are not directly comparable to analytical measures
computed from generating VAR/state-space parameters.
"""
from .discrete import (
    discrete_entropy,
    discrete_mutual_information,
    discrete_total_correlation,
    lempel_ziv_complexity,
)
from .emergence import emergence_from_observations
from .gaussian import local_gaussian_mutual_information
from .mvgc import spectral_mvgc as estimate_spectral_mvgc_from_observations
from .mvgc import temporal_mvgc as estimate_temporal_mvgc_from_observations

# Namespace-local compatibility aliases. At the package-level, temporal_mvgc and
# spectral_mvgc now denote the canonical model-based primary measures.
temporal_mvgc = estimate_temporal_mvgc_from_observations
spectral_mvgc = estimate_spectral_mvgc_from_observations

__all__ = [
    "estimate_temporal_mvgc_from_observations",
    "estimate_spectral_mvgc_from_observations",
    "temporal_mvgc",
    "spectral_mvgc",
    "emergence_from_observations",
    "local_gaussian_mutual_information",
    "discrete_entropy",
    "discrete_mutual_information",
    "discrete_total_correlation",
    "lempel_ziv_complexity",
]
