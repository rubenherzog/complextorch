API reference
=============

This reference indexes the public ComplexTorch API. Each section uses Sphinx
``autosummary`` to generate a dedicated autodoc page for every listed object.
Those pages are registered with the Python domain and, through
``sphinx.ext.viewcode``, expose a ``[source]`` link to the implementation.
Narrative documentation can therefore link to the same canonical objects with
``:func:``, ``:class:``, and related Sphinx roles.

VAR models and representations
------------------------------

.. autosummary::
   :toctree: generated

   ~complextorch.VAR
   ~complextorch.VARParameters
   ~complextorch.VARSystem
   ~complextorch.build_var_system
   ~complextorch.companion_matrix

State-space models
------------------

.. autosummary::
   :toctree: generated

   ~complextorch.StateSpaceModel
   ~complextorch.InnovationsStateSpace
   ~complextorch.N4SID
   ~complextorch.LarimoreStateSpace
   ~complextorch.LinearGaussianEM
   ~complextorch.kalman_filter
   ~complextorch.kalman_smoother
   ~complextorch.innovations_form
   ~complextorch.innovations_transfer_function
   ~complextorch.var_to_innovations_state_space
   ~complextorch.reduce_state_space
   ~complextorch.reduce_innovations_state_space
   ~complextorch.project_state_space

Model selection
---------------

.. autosummary::
   :toctree: generated

   ~complextorch.EpochTimeSeriesSplit
   ~complextorch.VAROrderSelectionIC
   ~complextorch.VAROrderSearchCV
   ~complextorch.VAROrderScore
   ~complextorch.VAROrderSearchResult
   ~complextorch.VARInformationCriteriaResult
   ~complextorch.StateSpaceOrderSelection
   ~complextorch.StateSpaceOrderSelectionResult
   ~complextorch.StateSpaceOrderSearchCV
   ~complextorch.StateSpaceOrderScore
   ~complextorch.StateSpaceOrderSearchResult

Control and Riccati methods
---------------------------

.. autosummary::
   :toctree: generated

   ~complextorch.solve_dare
   ~complextorch.solve_generalized_dare
   ~complextorch.ProjectionSearchResult
   ~complextorch.optimise_dynamical_dependence_projection

Dynamical-dependence optimization
---------------------------------

.. autosummary::
   :toctree: generated

   ~complextorch.optimise_dynamical_dependence
   ~complextorch.DDObjective
   ~complextorch.DDOptimizer
   ~complextorch.DDOptimizationResult
   ~complextorch.DDGradientSearchResult
   ~complextorch.dynamical_dependence
   ~complextorch.stochastic_interaction
   ~complextorch.innovations_proxy_sequence
   ~complextorch.optimise_dynamical_dependence_proxy
   ~complextorch.optimise_dynamical_dependence_spectral
   ~complextorch.orthonormalise_projection
   ~complextorch.proxy_dynamical_dependence
   ~complextorch.proxy_dynamical_dependence_gradient
   ~complextorch.spectral_dynamical_dependence
   ~complextorch.spectral_dynamical_dependence_gradient

Model-derived measures
----------------------

.. autosummary::
   :toctree: generated

   ~complextorch.ModelMeasureConfig
   ~complextorch.ModelMeasureContext
   ~complextorch.build_measure_context
   ~complextorch.compute_all_model_measures
   ~complextorch.model_autocovariances
   ~complextorch.temporal_mvgc
   ~complextorch.spectral_mvgc
   ~complextorch.gaussian_mutual_information_rate
   ~complextorch.gaussian_transfer_entropy_rate
   ~complextorch.gaussian_instantaneous_information_rate
   ~complextorch.spectral_gaussian_mutual_information_rate
   ~complextorch.spectral_gaussian_transfer_entropy_rate
   ~complextorch.o_information_rate
   ~complextorch.spectral_o_information_rate
   ~complextorch.delta_o_information_rate
   ~complextorch.spectral_delta_o_information_rate
   ~complextorch.partial_information_rate_decomposition
   ~complextorch.spectral_partial_information_rate_decomposition
   ~complextorch.PIRDResult
   ~complextorch.SpectralPIRDResult
   ~complextorch.partial_granger_causality_decomposition
   ~complextorch.spectral_partial_granger_causality_decomposition
   ~complextorch.PDGCResult
   ~complextorch.SpectralPDGCResult
   ~complextorch.hop_analysis
   ~complextorch.spectral_hop_analysis
   ~complextorch.HOPResult
   ~complextorch.SpectralHOPResult
   ~complextorch.phiid_from_model
   ~complextorch.phiid_redundancy_from_model
   ~complextorch.WhitenessResult
   ~complextorch.residual_whiteness
   ~complextorch.consistency
   ~complextorch.mvgc_pvalue
   ~complextorch.significance

Gaussian covariance primitives
------------------------------

These public primitives live in :mod:`complextorch.measures` and are useful
when the covariance matrix itself is the scientific input.

.. autosummary::
   :toctree: generated

   ~complextorch.measures.conditional_covariance
   ~complextorch.measures.gaussian_conditional_mutual_information
   ~complextorch.measures.total_correlation
   ~complextorch.measures.dual_total_correlation
   ~complextorch.measures.o_information
   ~complextorch.measures.s_information
   ~complextorch.measures.local_gaussian_mutual_information

Dynamics and additional measure primitives
------------------------------------------

.. autosummary::
   :toctree: generated

   ~complextorch.measures.entropy_rate
   ~complextorch.measures.predictive_information
   ~complextorch.measures.active_information_storage
   ~complextorch.measures.transfer_function
   ~complextorch.measures.inverse_transfer_function
   ~complextorch.measures.cross_spectral_density
   ~complextorch.measures.spectral_entropy
   ~complextorch.measures.state_space_temporal_mvgc
   ~complextorch.measures.pairwise_spectral_gc
   ~complextorch.measures.emergence_measures
   ~complextorch.measures.emergence_from_observations
   ~complextorch.measures.covariance_amplification
   ~complextorch.measures.dominant_timescale
   ~complextorch.measures.stability_margin
   ~complextorch.measures.gaussian_phiid_mmi
   ~complextorch.measures.gaussian_phiid_atoms

Simulation, spectra, and multiscale utilities
---------------------------------------------

.. autosummary::
   :toctree: generated

   ~complextorch.automatic_burnin
   ~complextorch.simulate_var
   ~complextorch.demo_var
   ~complextorch.random_stable_var
   ~complextorch.random_correlation_matrix
   ~complextorch.random_positive_definite_covariance
   ~complextorch.innovations_spectral_density
   ~complextorch.integrate_spectral_rate
   ~complextorch.downsample_innovations_state_space
   ~complextorch.varma_to_innovations_state_space

Adapters
--------

.. autosummary::
   :toctree: generated

   ~complextorch.from_complexbox_timeseries
   ~complextorch.to_complexbox_timeseries
   ~complextorch.from_complexbox_var
   ~complextorch.to_complexbox_var
