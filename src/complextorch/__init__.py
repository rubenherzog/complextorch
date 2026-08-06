"""Torch-first batched linear-dynamics inference and complexity measures."""
from .adapters import from_complexbox_timeseries, from_complexbox_var, to_complexbox_timeseries, to_complexbox_var
from .representations import LinearDynamicalSystem, VARSystem, build_var_system, companion_matrix
from .selection import EpochTimeSeriesSplit, VAROrderSearchCV, VAROrderSearchResult, VAROrderSelectionIC, VARInformationCriteriaResult
from .model_order import LarimoreStateSpaceOrder, LarimoreStateSpaceOrderResult, bauer_svc, larimore_state_space_order
from .simulate import demo_var, random_stable_var, simulate_var, automatic_burnin, random_correlation_matrix, random_positive_definite_covariance
from .var import VAR, VARParameters
from .control import solve_dare, solve_generalized_dare, innovations_form, InnovationsStateSpace, var_to_innovations_state_space, reduce_innovations_state_space, innovations_transfer_function, reduce_state_space, project_state_space, dynamical_dependence, stochastic_interaction, optimise_dynamical_dependence_projection, ProjectionSearchResult
from .state_space import kalman_filter, kalman_smoother, N4SID, LinearGaussianEM
from .measures.primary import ModelMeasureConfig, ModelMeasureContext, build_measure_context, compute_all_model_measures, model_autocovariances, phiid_from_model, spectral_mvgc, temporal_mvgc
from .measures.secondary import WhitenessResult, consistency, residual_whiteness, mvgc_pvalue, significance

__all__ = [
    "VAR", "VARParameters", "VARSystem", "LinearDynamicalSystem", "build_var_system", "companion_matrix",
    "EpochTimeSeriesSplit", "VAROrderSearchCV", "VAROrderSearchResult", "VAROrderSelectionIC", "VARInformationCriteriaResult",
    "LarimoreStateSpaceOrder", "LarimoreStateSpaceOrderResult", "bauer_svc", "larimore_state_space_order",
    "simulate_var", "automatic_burnin", "random_stable_var", "random_correlation_matrix", "random_positive_definite_covariance", "demo_var",
    "from_complexbox_timeseries", "to_complexbox_timeseries", "from_complexbox_var", "to_complexbox_var",
    "solve_dare", "solve_generalized_dare", "innovations_form", "InnovationsStateSpace", "var_to_innovations_state_space", "reduce_innovations_state_space", "innovations_transfer_function", "reduce_state_space", "project_state_space", "dynamical_dependence", "stochastic_interaction", "optimise_dynamical_dependence_projection", "ProjectionSearchResult",
    "kalman_filter", "kalman_smoother", "N4SID", "LinearGaussianEM",
    "ModelMeasureConfig", "ModelMeasureContext", "build_measure_context", "compute_all_model_measures", "model_autocovariances", "phiid_from_model", "temporal_mvgc", "spectral_mvgc",
    "WhitenessResult", "consistency", "residual_whiteness", "mvgc_pvalue", "significance",
]
__version__ = "0.7.0"
