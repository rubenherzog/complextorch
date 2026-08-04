"""Torch-first batched linear-dynamics inference and complexity measures."""
from .adapters import from_complexbox_timeseries, from_complexbox_var, to_complexbox_timeseries, to_complexbox_var
from .representations import LinearDynamicalSystem, VARSystem, build_var_system, companion_matrix
from .selection import EpochTimeSeriesSplit, VAROrderSearchCV, VAROrderSearchResult
from .simulate import demo_var, random_stable_var, simulate_var
from .var import VAR, VARParameters
from .control import solve_dare, solve_generalized_dare, innovations_form, InnovationsStateSpace, var_to_innovations_state_space, reduce_innovations_state_space, innovations_transfer_function, reduce_state_space, project_state_space, dynamical_dependence, stochastic_interaction, optimise_dynamical_dependence_projection, ProjectionSearchResult
from .state_space import kalman_filter, kalman_smoother, N4SID, LinearGaussianEM
from .measures.primary import (
    ModelMeasureConfig,
    ModelMeasureContext,
    build_measure_context,
    compute_all_model_measures,
    model_autocovariances,
    phiid_from_model,
    spectral_mvgc,
    temporal_mvgc,
)

__all__ = [
    "VAR", "VARParameters", "VARSystem", "LinearDynamicalSystem", "build_var_system", "companion_matrix",
    "EpochTimeSeriesSplit", "VAROrderSearchCV", "VAROrderSearchResult", "simulate_var", "random_stable_var", "demo_var",
    "from_complexbox_timeseries", "to_complexbox_timeseries", "from_complexbox_var", "to_complexbox_var",
    "solve_dare", "solve_generalized_dare", "innovations_form", "InnovationsStateSpace",
    "var_to_innovations_state_space", "reduce_innovations_state_space", "innovations_transfer_function",
    "reduce_state_space", "project_state_space", "dynamical_dependence", "stochastic_interaction",
    "optimise_dynamical_dependence_projection", "ProjectionSearchResult",
    "kalman_filter", "kalman_smoother", "N4SID", "LinearGaussianEM",
    "ModelMeasureConfig", "ModelMeasureContext", "build_measure_context",
    "compute_all_model_measures", "model_autocovariances", "phiid_from_model",
    "temporal_mvgc", "spectral_mvgc",
]
__version__ = "0.4.1"
