API reference
=============

Every public object below links explicitly to its dedicated API page. Each API
page is generated from the live package docstring and includes a ``[source]``
link, provided by :mod:`sphinx.ext.viewcode`, to the corresponding implementation.

VAR models and representations
------------------------------

- :doc:`VAR <generated/complextorch.VAR>`
- :doc:`VARParameters <generated/complextorch.VARParameters>`
- :doc:`VARSystem <generated/complextorch.VARSystem>`
- :doc:`build_var_system <generated/complextorch.build_var_system>`
- :doc:`companion_matrix <generated/complextorch.companion_matrix>`

State-space models
------------------

- :doc:`StateSpaceModel <generated/complextorch.StateSpaceModel>`
- :doc:`InnovationsStateSpace <generated/complextorch.InnovationsStateSpace>`
- :doc:`N4SID <generated/complextorch.N4SID>`
- :doc:`LarimoreStateSpace <generated/complextorch.LarimoreStateSpace>`
- :doc:`LinearGaussianEM <generated/complextorch.LinearGaussianEM>`
- :doc:`kalman_filter <generated/complextorch.kalman_filter>`
- :doc:`kalman_smoother <generated/complextorch.kalman_smoother>`
- :doc:`innovations_form <generated/complextorch.innovations_form>`
- :doc:`innovations_transfer_function <generated/complextorch.innovations_transfer_function>`
- :doc:`var_to_innovations_state_space <generated/complextorch.var_to_innovations_state_space>`
- :doc:`reduce_state_space <generated/complextorch.reduce_state_space>`
- :doc:`reduce_innovations_state_space <generated/complextorch.reduce_innovations_state_space>`
- :doc:`project_state_space <generated/complextorch.project_state_space>`

Model selection
---------------

- :doc:`EpochTimeSeriesSplit <generated/complextorch.EpochTimeSeriesSplit>`
- :doc:`VAROrderSelectionIC <generated/complextorch.VAROrderSelectionIC>`
- :doc:`VAROrderSearchCV <generated/complextorch.VAROrderSearchCV>`
- :doc:`VAROrderScore <generated/complextorch.VAROrderScore>`
- :doc:`VAROrderSearchResult <generated/complextorch.VAROrderSearchResult>`
- :doc:`VARInformationCriteriaResult <generated/complextorch.VARInformationCriteriaResult>`
- :doc:`StateSpaceOrderSelection <generated/complextorch.StateSpaceOrderSelection>`
- :doc:`StateSpaceOrderSelectionResult <generated/complextorch.StateSpaceOrderSelectionResult>`
- :doc:`StateSpaceOrderSearchCV <generated/complextorch.StateSpaceOrderSearchCV>`
- :doc:`StateSpaceOrderScore <generated/complextorch.StateSpaceOrderScore>`
- :doc:`StateSpaceOrderSearchResult <generated/complextorch.StateSpaceOrderSearchResult>`

Control and Riccati methods
---------------------------

- :doc:`solve_dare <generated/complextorch.solve_dare>`
- :doc:`solve_generalized_dare <generated/complextorch.solve_generalized_dare>`
- :doc:`ProjectionSearchResult <generated/complextorch.ProjectionSearchResult>`
- :doc:`optimise_dynamical_dependence_projection <generated/complextorch.optimise_dynamical_dependence_projection>`

Dynamical-dependence optimization
---------------------------------

:func:`~complextorch.optimise_dynamical_dependence` uses the canonical staged
SSDI workflow when ``objective=None`` and returns
:class:`~complextorch.DDSSDIOptimizationResult`. Explicit single-stage proxy or
spectral optimization returns :class:`~complextorch.DDOptimizationResult`.

- :doc:`optimise_dynamical_dependence <generated/complextorch.optimise_dynamical_dependence>`
- :doc:`DDObjective <generated/complextorch.DDObjective>`
- :doc:`DDOptimizer <generated/complextorch.DDOptimizer>`
- :doc:`DDOptimizationResult <generated/complextorch.DDOptimizationResult>`
- :doc:`DDSSDIOptimizationResult <generated/complextorch.DDSSDIOptimizationResult>`
- :doc:`DDGradientSearchResult <generated/complextorch.DDGradientSearchResult>`
- :doc:`dynamical_dependence <generated/complextorch.dynamical_dependence>`
- :doc:`stochastic_interaction <generated/complextorch.stochastic_interaction>`
- :doc:`innovations_proxy_sequence <generated/complextorch.innovations_proxy_sequence>`
- :doc:`optimise_dynamical_dependence_proxy <generated/complextorch.optimise_dynamical_dependence_proxy>`
- :doc:`optimise_dynamical_dependence_spectral <generated/complextorch.optimise_dynamical_dependence_spectral>`
- :doc:`orthonormalise_projection <generated/complextorch.orthonormalise_projection>`
- :doc:`proxy_dynamical_dependence <generated/complextorch.proxy_dynamical_dependence>`
- :doc:`proxy_dynamical_dependence_gradient <generated/complextorch.proxy_dynamical_dependence_gradient>`
- :doc:`spectral_dynamical_dependence <generated/complextorch.spectral_dynamical_dependence>`
- :doc:`spectral_dynamical_dependence_gradient <generated/complextorch.spectral_dynamical_dependence_gradient>`

Model-derived measures
----------------------

- :doc:`ModelMeasureConfig <generated/complextorch.ModelMeasureConfig>`
- :doc:`ModelMeasureContext <generated/complextorch.ModelMeasureContext>`
- :doc:`build_measure_context <generated/complextorch.build_measure_context>`
- :doc:`compute_all_model_measures <generated/complextorch.compute_all_model_measures>`
- :doc:`model_autocovariances <generated/complextorch.model_autocovariances>`
- :doc:`temporal_mvgc <generated/complextorch.temporal_mvgc>`
- :doc:`spectral_mvgc <generated/complextorch.spectral_mvgc>`
- :doc:`gaussian_mutual_information_rate <generated/complextorch.gaussian_mutual_information_rate>`
- :doc:`gaussian_transfer_entropy_rate <generated/complextorch.gaussian_transfer_entropy_rate>`
- :doc:`gaussian_instantaneous_information_rate <generated/complextorch.gaussian_instantaneous_information_rate>`
- :doc:`spectral_gaussian_mutual_information_rate <generated/complextorch.spectral_gaussian_mutual_information_rate>`
- :doc:`spectral_gaussian_transfer_entropy_rate <generated/complextorch.spectral_gaussian_transfer_entropy_rate>`
- :doc:`o_information_rate <generated/complextorch.o_information_rate>`
- :doc:`spectral_o_information_rate <generated/complextorch.spectral_o_information_rate>`
- :doc:`delta_o_information_rate <generated/complextorch.delta_o_information_rate>`
- :doc:`spectral_delta_o_information_rate <generated/complextorch.spectral_delta_o_information_rate>`
- :doc:`partial_information_rate_decomposition <generated/complextorch.partial_information_rate_decomposition>`
- :doc:`spectral_partial_information_rate_decomposition <generated/complextorch.spectral_partial_information_rate_decomposition>`
- :doc:`PIRDResult <generated/complextorch.PIRDResult>`
- :doc:`SpectralPIRDResult <generated/complextorch.SpectralPIRDResult>`
- :doc:`partial_granger_causality_decomposition <generated/complextorch.partial_granger_causality_decomposition>`
- :doc:`spectral_partial_granger_causality_decomposition <generated/complextorch.spectral_partial_granger_causality_decomposition>`
- :doc:`PDGCResult <generated/complextorch.PDGCResult>`
- :doc:`SpectralPDGCResult <generated/complextorch.SpectralPDGCResult>`
- :doc:`hop_analysis <generated/complextorch.hop_analysis>`
- :doc:`spectral_hop_analysis <generated/complextorch.spectral_hop_analysis>`
- :doc:`HOPResult <generated/complextorch.HOPResult>`
- :doc:`SpectralHOPResult <generated/complextorch.SpectralHOPResult>`
- :doc:`phiid_from_model <generated/complextorch.phiid_from_model>`
- :doc:`phiid_redundancy_from_model <generated/complextorch.phiid_redundancy_from_model>`
- :doc:`WhitenessResult <generated/complextorch.WhitenessResult>`
- :doc:`residual_whiteness <generated/complextorch.residual_whiteness>`
- :doc:`consistency <generated/complextorch.consistency>`
- :doc:`mvgc_pvalue <generated/complextorch.mvgc_pvalue>`
- :doc:`significance <generated/complextorch.significance>`

Gaussian covariance primitives
------------------------------

These public primitives live in :mod:`complextorch.measures` and are useful
when the covariance matrix itself is the scientific input.

- :doc:`conditional_covariance <generated/complextorch.measures.conditional_covariance>`
- :doc:`gaussian_conditional_mutual_information <generated/complextorch.measures.gaussian_conditional_mutual_information>`
- :doc:`total_correlation <generated/complextorch.measures.total_correlation>`
- :doc:`dual_total_correlation <generated/complextorch.measures.dual_total_correlation>`
- :doc:`o_information <generated/complextorch.measures.o_information>`
- :doc:`s_information <generated/complextorch.measures.s_information>`
- :doc:`local_gaussian_mutual_information <generated/complextorch.measures.local_gaussian_mutual_information>`

Dynamics and additional measure primitives
------------------------------------------

- :doc:`entropy_rate <generated/complextorch.measures.entropy_rate>`
- :doc:`predictive_information <generated/complextorch.measures.predictive_information>`
- :doc:`active_information_storage <generated/complextorch.measures.active_information_storage>`
- :doc:`transfer_function <generated/complextorch.measures.transfer_function>`
- :doc:`inverse_transfer_function <generated/complextorch.measures.inverse_transfer_function>`
- :doc:`cross_spectral_density <generated/complextorch.measures.cross_spectral_density>`
- :doc:`spectral_entropy <generated/complextorch.measures.spectral_entropy>`
- :doc:`state_space_temporal_mvgc <generated/complextorch.measures.state_space_temporal_mvgc>`
- :doc:`pairwise_spectral_gc <generated/complextorch.measures.pairwise_spectral_gc>`
- :doc:`emergence_measures <generated/complextorch.measures.emergence_measures>`
- :doc:`emergence_from_observations <generated/complextorch.measures.emergence_from_observations>`
- :doc:`covariance_amplification <generated/complextorch.measures.covariance_amplification>`
- :doc:`dominant_timescale <generated/complextorch.measures.dominant_timescale>`
- :doc:`stability_margin <generated/complextorch.measures.stability_margin>`
- :doc:`gaussian_phiid_mmi <generated/complextorch.measures.gaussian_phiid_mmi>`
- :doc:`gaussian_phiid_atoms <generated/complextorch.measures.gaussian_phiid_atoms>`

Simulation, spectra, and multiscale utilities
---------------------------------------------

- :doc:`automatic_burnin <generated/complextorch.automatic_burnin>`
- :doc:`simulate_var <generated/complextorch.simulate_var>`
- :doc:`demo_var <generated/complextorch.demo_var>`
- :doc:`random_stable_var <generated/complextorch.random_stable_var>`
- :doc:`random_correlation_matrix <generated/complextorch.random_correlation_matrix>`
- :doc:`random_positive_definite_covariance <generated/complextorch.random_positive_definite_covariance>`
- :doc:`innovations_spectral_density <generated/complextorch.innovations_spectral_density>`
- :doc:`integrate_spectral_rate <generated/complextorch.integrate_spectral_rate>`
- :doc:`downsample_innovations_state_space <generated/complextorch.downsample_innovations_state_space>`
- :doc:`varma_to_innovations_state_space <generated/complextorch.varma_to_innovations_state_space>`

Adapters
--------

- :doc:`from_complexbox_timeseries <generated/complextorch.from_complexbox_timeseries>`
- :doc:`to_complexbox_timeseries <generated/complextorch.to_complexbox_timeseries>`
- :doc:`from_complexbox_var <generated/complextorch.from_complexbox_var>`
- :doc:`to_complexbox_var <generated/complextorch.to_complexbox_var>`
