"""Machine-readable catalogue of model-first and empirical measure tiers.

Notes
-----
The registry declares which measures belong to the analytical primary tier and
which require observations or sample estimators.

Notes
-----
The registry declares which measures belong to the analytical primary tier and
which require observations or sample estimators.

Notes
-----
The registry declares which measures belong to the analytical primary tier and
which require observations or sample estimators.

Notes
-----
The registry declares which measures belong to the analytical primary tier and
which require observations or sample estimators.
"""
MEASURE_REGISTRY = {
    "gaussian_entropy": {"domain": "model_covariance", "analytic": True, "tier": "primary", "model_entrypoint": "gaussian_measures_from_model"},
    "mutual_information": {"domain": "model_covariance", "analytic": True, "tier": "primary", "model_entrypoint": "gaussian_measures_from_model"},
    "conditional_mutual_information": {"domain": "model_covariance", "analytic": True, "tier": "primary_primitive"},
    "conditional_covariance": {"domain": "model_covariance", "analytic": True, "tier": "primary_primitive"},
    "total_correlation": {"domain": "model_covariance", "analytic": True, "tier": "primary", "model_entrypoint": "gaussian_measures_from_model"},
    "dual_total_correlation": {"domain": "model_covariance", "analytic": True, "tier": "primary", "model_entrypoint": "gaussian_measures_from_model"},
    "o_information": {"domain": "model_covariance", "analytic": True, "tier": "primary", "model_entrypoint": "gaussian_measures_from_model"},
    "s_information": {"domain": "model_covariance", "analytic": True, "tier": "primary", "model_entrypoint": "gaussian_measures_from_model"},
    "autocovariances": {"domain": "var_or_state_space", "analytic": True, "tier": "primary", "shared_core": "model_measure_context"},
    "entropy_rate": {"domain": "var_system", "analytic": True, "tier": "primary"},
    "predictive_information": {"domain": "var_system", "analytic": True, "tier": "primary"},
    "active_information_storage": {"domain": "var_system", "analytic": True, "tier": "primary"},
    "transfer_function": {"domain": "var_system_frequency", "analytic": True, "tier": "primary_primitive"},
    "inverse_transfer_function": {"domain": "var_system_frequency", "analytic": True, "tier": "primary_primitive"},
    "cross_spectral_density": {"domain": "var_system_frequency", "analytic": True, "tier": "primary"},
    "spectral_entropy": {"domain": "var_system_frequency", "analytic": True, "tier": "primary"},
    "spectral_radius": {"domain": "var_system", "analytic": True, "tier": "primary"},
    "stability_margin": {"domain": "var_system", "analytic": True, "tier": "primary"},
    "dominant_timescale": {"domain": "var_system", "analytic": True, "tier": "primary"},
    "covariance_amplification": {"domain": "var_system", "analytic": True, "tier": "primary"},
    "psi": {"domain": "var_system_macro_projection", "analytic": True, "tier": "primary"},
    "delta": {"domain": "var_system_macro_projection", "analytic": True, "tier": "primary"},
    "gamma": {"domain": "var_system_macro_projection", "analytic": True, "tier": "primary"},
    "cmem1_total": {"domain": "var_system", "analytic": True, "tier": "primary", "shared_core": "model_measure_context"},
    "cmem3_total": {"domain": "var_system", "analytic": True, "tier": "primary", "shared_core": "model_measure_context"},
    "cmem1_curve": {"domain": "var_system_delay", "analytic": True, "tier": "primary", "default_delay": 1, "shared_core": "model_measure_context"},
    "cmem3_curve": {"domain": "var_system_delay", "analytic": True, "tier": "primary", "default_delay": 1, "shared_core": "model_measure_context"},
    "cmem3_lag": {"domain": "var_system", "analytic": True, "tier": "primary", "shared_core": "model_measure_context"},
    "temporal_mvgc": {"domain": "var_or_state_space", "analytic": True, "tier": "primary", "shared_core": "generalized_dare_marginals"},
    "spectral_mvgc": {"domain": "var_or_state_space_frequency", "analytic": True, "tier": "primary", "shared_core": "generalized_dare_marginals"},
    "state_space_temporal_mvgc": {"domain": "var_or_innovations_state_space", "analytic": True, "tier": "primary_primitive", "shared_core": "generalized_dare_marginals"},
    "state_space_spectral_mvgc": {"domain": "var_or_innovations_state_space_frequency", "analytic": True, "tier": "primary_primitive", "shared_core": "generalized_dare_marginals"},
    "integrate_spectral_mvgc": {"domain": "frequency_curve", "analytic": True, "tier": "primary_primitive"},
    "pairwise_spectral_gc": {"domain": "var_system_frequency", "analytic": True, "tier": "primary"},
    "dynamical_dependence": {"domain": "state_space", "analytic": True, "tier": "primary", "shared_core": "innovations_form"},
    "ssdi": {"domain": "state_space_partition", "analytic": True, "tier": "primary", "shared_core": "reduced_state_space"},
    "gaussian_phiid_atoms": {"domain": "var_or_state_space_delay", "analytic": True, "tier": "primary", "atoms": 16, "default_delay": 1, "model_entrypoint": "phiid_from_model", "shared_core": "model_measure_context"},
    "estimate_temporal_mvgc_from_observations": {"domain": "observations", "analytic": False, "tier": "secondary", "shared_core": "nested_var_models"},
    "estimate_spectral_mvgc_from_observations": {"domain": "observations_frequency", "analytic": False, "tier": "secondary", "shared_core": "nested_var_models"},
    "emergence_from_observations": {"domain": "observations", "analytic": False, "tier": "secondary"},
    "local_gaussian_mi": {"domain": "observations", "analytic": False, "tier": "secondary"},
    "discrete_entropy": {"domain": "discrete_observations", "analytic": False, "tier": "secondary"},
    "discrete_mutual_information": {"domain": "discrete_observations", "analytic": False, "tier": "secondary"},
    "discrete_total_correlation": {"domain": "discrete_observations", "analytic": False, "tier": "secondary"},
    "lempel_ziv_complexity": {"domain": "discrete_sequence", "analytic": False, "tier": "secondary"},
}

PRIMARY_MEASURES = {
    name: metadata
    for name, metadata in MEASURE_REGISTRY.items()
    if metadata["tier"].startswith("primary")
}
SECONDARY_MEASURES = {
    name: metadata
    for name, metadata in MEASURE_REGISTRY.items()
    if metadata["tier"] == "secondary"
}
