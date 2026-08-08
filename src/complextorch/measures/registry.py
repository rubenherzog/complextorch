"""Registry separating analytical primary and empirical secondary measures.

The registry is the inventory behind the model-first contract.  New analytical
families must be registered here when they enter the public API so the catalog
cannot silently drift away from :func:`compute_all_model_measures`.
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
    "autocovariances": {"domain": "var_or_state_space", "analytic": True, "tier": "primary_primitive", "shared_core": "model_measure_context"},
    "entropy_rate": {"domain": "var_or_state_space", "analytic": True, "tier": "primary"},
    "marginal_entropy_rate": {"domain": "var_or_state_space", "analytic": True, "tier": "primary", "shared_core": "generalized_dare_marginals", "model_entrypoint": "marginal_entropy_rate"},
    "predictive_information": {"domain": "var_or_state_space", "analytic": True, "tier": "primary"},
    "active_information_storage": {"domain": "var_or_state_space_delay", "analytic": True, "tier": "primary"},
    "transfer_function": {"domain": "innovations_state_space_frequency", "analytic": True, "tier": "primary_primitive", "shared_core": "model_measure_context"},
    "inverse_transfer_function": {"domain": "innovations_state_space_frequency", "analytic": True, "tier": "primary_primitive", "shared_core": "model_measure_context"},
    "cross_spectral_density": {"domain": "var_or_state_space_frequency", "analytic": True, "tier": "primary_primitive", "shared_core": "model_measure_context"},
    "spectral_entropy": {"domain": "var_or_state_space_frequency", "analytic": True, "tier": "primary"},
    "spectral_entropy_rate": {"domain": "var_or_state_space_frequency", "analytic": True, "tier": "primary", "model_entrypoint": "spectral_entropy_rate"},
    "spectral_radius": {"domain": "var_or_state_space", "analytic": True, "tier": "primary"},
    "stability_margin": {"domain": "var_or_state_space", "analytic": True, "tier": "primary"},
    "dominant_timescale": {"domain": "var_or_state_space", "analytic": True, "tier": "primary"},
    "covariance_amplification": {"domain": "var_or_state_space", "analytic": True, "tier": "primary"},
    "psi": {"domain": "var_or_state_space_macro_projection", "analytic": True, "tier": "primary"},
    "delta": {"domain": "var_or_state_space_macro_projection", "analytic": True, "tier": "primary"},
    "gamma": {"domain": "var_or_state_space_macro_projection", "analytic": True, "tier": "primary"},
    "cmem1_total": {"domain": "var_or_state_space", "analytic": True, "tier": "primary", "shared_core": "model_measure_context"},
    "cmem3_total": {"domain": "var_or_state_space", "analytic": True, "tier": "primary", "shared_core": "model_measure_context"},
    "cmem1_curve": {"domain": "var_or_state_space_delay", "analytic": True, "tier": "primary", "default_delay": 1, "shared_core": "model_measure_context"},
    "cmem3_curve": {"domain": "var_or_state_space_delay", "analytic": True, "tier": "primary", "default_delay": 1, "shared_core": "model_measure_context"},
    "cmem3_lag": {"domain": "var_or_state_space", "analytic": True, "tier": "primary", "shared_core": "model_measure_context"},
    "temporal_mvgc": {"domain": "var_or_state_space", "analytic": True, "tier": "primary", "shared_core": "generalized_dare_marginals"},
    "spectral_mvgc": {"domain": "var_or_state_space_frequency", "analytic": True, "tier": "primary", "shared_core": "generalized_dare_marginals"},
    "state_space_temporal_mvgc": {"domain": "innovations_state_space", "analytic": True, "tier": "primary_primitive", "shared_core": "generalized_dare_marginals"},
    "state_space_spectral_mvgc": {"domain": "innovations_state_space_frequency", "analytic": True, "tier": "primary_primitive", "shared_core": "generalized_dare_marginals"},
    "integrate_spectral_mvgc": {"domain": "frequency_curve", "analytic": True, "tier": "primary_primitive"},
    "pairwise_spectral_gc": {"domain": "var_or_state_space_frequency", "analytic": True, "tier": "primary"},
    "gaussian_mutual_information_rate": {"domain": "innovations_state_space_groups", "analytic": True, "tier": "primary", "shared_core": "generalized_dare_marginals"},
    "gaussian_transfer_entropy_rate": {"domain": "innovations_state_space_groups", "analytic": True, "tier": "primary", "shared_core": "generalized_dare_marginals"},
    "gaussian_instantaneous_information_rate": {"domain": "innovations_state_space_groups", "analytic": True, "tier": "primary", "shared_core": "generalized_dare_marginals"},
    "spectral_gaussian_mutual_information_rate": {"domain": "innovations_state_space_groups_frequency", "analytic": True, "tier": "primary", "shared_core": "generalized_dare_marginals"},
    "spectral_gaussian_transfer_entropy_rate": {"domain": "innovations_state_space_groups_frequency", "analytic": True, "tier": "primary", "shared_core": "generalized_dare_marginals"},
    "o_information_rate": {"domain": "innovations_state_space_groups", "analytic": True, "tier": "primary", "shared_core": "generalized_dare_marginals"},
    "spectral_o_information_rate": {"domain": "innovations_state_space_groups_frequency", "analytic": True, "tier": "primary", "shared_core": "generalized_dare_marginals"},
    "delta_o_information_rate": {"domain": "innovations_state_space_groups", "analytic": True, "tier": "primary", "shared_core": "generalized_dare_marginals"},
    "spectral_delta_o_information_rate": {"domain": "innovations_state_space_groups_frequency", "analytic": True, "tier": "primary", "shared_core": "generalized_dare_marginals"},
    "dynamical_dependence": {"domain": "var_or_state_space_macro_projection", "analytic": True, "tier": "primary", "shared_core": "innovations_form"},
    "stochastic_interaction": {"domain": "var_or_state_space_partition", "analytic": True, "tier": "primary", "shared_core": "reduced_state_space"},
    "gaussian_phiid_atoms": {"domain": "var_or_state_space_delay", "analytic": True, "tier": "primary", "atoms": 16, "default_delay": 1, "model_entrypoint": "phiid_from_model", "shared_core": "model_measure_context"},
    "phiid_redundancy_mmi": {"domain": "var_or_state_space_delay", "analytic": True, "tier": "primary", "atoms": 16, "model_entrypoint": "phiid_redundancy_from_model", "shared_core": "model_measure_context"},
    "phiid_redundancy_ccs": {"domain": "var_or_state_space_delay", "analytic": True, "tier": "primary", "atoms": 16, "model_entrypoint": "phiid_redundancy_from_model", "shared_core": "model_measure_context"},
    "phiid_redundancy_idep_a": {"domain": "var_or_state_space_delay", "analytic": True, "tier": "primary", "atoms": 16, "model_entrypoint": "phiid_redundancy_from_model", "shared_core": "model_measure_context"},
    "phiid_redundancy_idep_b": {"domain": "var_or_state_space_delay", "analytic": True, "tier": "primary", "atoms": 16, "model_entrypoint": "phiid_redundancy_from_model", "shared_core": "model_measure_context"},
    "partial_information_rate_decomposition": {"domain": "innovations_state_space_sources_target_frequency", "analytic": True, "tier": "primary", "shared_core": "hop"},
    "spectral_partial_information_rate_decomposition": {"domain": "innovations_state_space_sources_target_frequency", "analytic": True, "tier": "primary", "shared_core": "hop"},
    "partial_granger_causality_decomposition": {"domain": "innovations_state_space_sources_target_frequency", "analytic": True, "tier": "primary", "shared_core": "hop"},
    "spectral_partial_granger_causality_decomposition": {"domain": "innovations_state_space_sources_target_frequency", "analytic": True, "tier": "primary", "shared_core": "hop"},
    "hop_analysis": {"domain": "innovations_state_space_sources_target_frequency", "analytic": True, "tier": "primary", "shared_core": "hop"},
    "spectral_hop_analysis": {"domain": "innovations_state_space_sources_target_frequency", "analytic": True, "tier": "primary", "shared_core": "hop"},
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
