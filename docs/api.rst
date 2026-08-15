API reference
=============

This page is a compact index of the public ComplexTorch API. Each row shows the
public name, a short description derived from the live docstring, a link to the
full generated documentation, and a direct link to the corresponding source
code. The source link is pinned to the exact documentation build commit when
Read the Docs provides it.

VAR models and representations
------------------------------

.. api-table::

   complextorch.VAR
   complextorch.VARParameters
   complextorch.VARSystem
   complextorch.build_var_system
   complextorch.companion_matrix

State-space models
------------------

.. api-table::

   complextorch.StateSpaceModel
   complextorch.InnovationsStateSpace
   complextorch.N4SID
   complextorch.LarimoreStateSpace
   complextorch.LinearGaussianEM
   complextorch.kalman_filter
   complextorch.kalman_smoother
   complextorch.innovations_form
   complextorch.innovations_transfer_function
   complextorch.var_to_innovations_state_space
   complextorch.reduce_state_space
   complextorch.reduce_innovations_state_space
   complextorch.project_state_space

Canonical model transformations
-------------------------------

:func:`~complextorch.as_innovations_state_space` provides the public common
conversion for supported canonical dynamical models.  Model transformations
such as :func:`~complextorch.scale_dynamics` act on this common representation
without refitting observations.

.. api-table::

   complextorch.as_innovations_state_space
   complextorch.scale_dynamics

Model selection
---------------

.. api-table::

   complextorch.EpochTimeSeriesSplit
   complextorch.VAROrderSelectionIC
   complextorch.VAROrderSearchCV
   complextorch.VAROrderScore
   complextorch.VAROrderSearchResult
   complextorch.VARInformationCriteriaResult
   complextorch.StateSpaceOrderSelection
   complextorch.StateSpaceOrderSelectionResult
   complextorch.StateSpaceOrderSearchCV
   complextorch.StateSpaceOrderScore
   complextorch.StateSpaceOrderSearchResult

Control and Riccati methods
---------------------------

.. api-table::

   complextorch.solve_dare
   complextorch.solve_generalized_dare
   complextorch.ProjectionSearchResult
   complextorch.optimise_dynamical_dependence_projection

Dynamical dependence and optimization
-------------------------------------

:func:`~complextorch.dynamical_dependence` evaluates DD for a supplied
projection. :func:`~complextorch.optimise_dynamical_dependence` searches over
projection subspaces using the canonical staged SSDI workflow when
``objective=None`` and returns :class:`~complextorch.DDSSDIOptimizationResult`.
The default numerical step policy is ``optimizer="adaptive"``; ``"armijo"``
runs the same scientific proxy--cluster--spectral workflow with Riemannian
Armijo backtracking. Explicit single-stage proxy or spectral optimization
returns :class:`~complextorch.DDOptimizationResult`.

.. api-table::

   complextorch.optimise_dynamical_dependence
   complextorch.DDObjective
   complextorch.DDOptimizer
   complextorch.DDOptimizationResult
   complextorch.DDSSDIOptimizationResult
   complextorch.DDGradientSearchResult
   complextorch.dynamical_dependence
   complextorch.stochastic_interaction
   complextorch.innovations_proxy_sequence
   complextorch.optimise_dynamical_dependence_proxy
   complextorch.optimise_dynamical_dependence_spectral
   complextorch.orthonormalise_projection
   complextorch.proxy_dynamical_dependence
   complextorch.proxy_dynamical_dependence_gradient
   complextorch.spectral_dynamical_dependence
   complextorch.spectral_dynamical_dependence_gradient

Model-derived measures
----------------------

.. api-table::

   complextorch.ModelMeasureConfig
   complextorch.ModelMeasureContext
   complextorch.build_measure_context
   complextorch.compute_all_model_measures
   complextorch.model_autocovariances
   complextorch.temporal_mvgc
   complextorch.spectral_mvgc
   complextorch.gaussian_mutual_information_rate
   complextorch.gaussian_transfer_entropy_rate
   complextorch.gaussian_instantaneous_information_rate
   complextorch.spectral_gaussian_mutual_information_rate
   complextorch.spectral_gaussian_transfer_entropy_rate
   complextorch.o_information_rate
   complextorch.spectral_o_information_rate
   complextorch.delta_o_information_rate
   complextorch.spectral_delta_o_information_rate
   complextorch.partial_information_rate_decomposition
   complextorch.spectral_partial_information_rate_decomposition
   complextorch.PIRDResult
   complextorch.SpectralPIRDResult
   complextorch.partial_granger_causality_decomposition
   complextorch.spectral_partial_granger_causality_decomposition
   complextorch.PDGCResult
   complextorch.SpectralPDGCResult
   complextorch.hop_analysis
   complextorch.spectral_hop_analysis
   complextorch.HOPResult
   complextorch.SpectralHOPResult
   complextorch.phiid_from_model
   complextorch.phiid_redundancy_from_model
   complextorch.WhitenessResult
   complextorch.residual_whiteness
   complextorch.consistency
   complextorch.mvgc_pvalue
   complextorch.significance

Fit diagnostics
---------------

In-sample and out-of-sample evaluation are exposed through the same public
:func:`~complextorch.fit_diagnostics` entry point and share the
:func:`~complextorch.innovation_diagnostics` statistical core.

.. api-table::

   complextorch.FitDiagnostics
   complextorch.fit_diagnostics
   complextorch.innovation_diagnostics

Resampling inference and confidence intervals
---------------------------------------------

The confidence-interval API fits a fixed-order VAR, generates one shared
resampling ensemble, and evaluates all requested compatible analytical measures
on that ensemble. Residual bootstrap and Gaussian parametric Monte Carlo are
available. Dynamical-dependence intervals always use the supplied fixed
projection or fixed batch of projections; projection optimization is never
repeated inside resampling.

.. api-table::

   complextorch.measure_confidence_intervals
   complextorch.InferenceMeasureConfig
   complextorch.ConfidenceIntervalResult
   complextorch.MeasureInterval

Gaussian covariance primitives
------------------------------

These public primitives live in :mod:`complextorch.measures` and are useful
when the covariance matrix itself is the scientific input.

.. api-table::

   complextorch.measures.conditional_covariance
   complextorch.measures.gaussian_conditional_mutual_information
   complextorch.measures.total_correlation
   complextorch.measures.dual_total_correlation
   complextorch.measures.o_information
   complextorch.measures.s_information
   complextorch.measures.local_gaussian_mutual_information

Dynamics and additional measure primitives
------------------------------------------

.. api-table::

   complextorch.measures.entropy_rate
   complextorch.measures.predictive_information
   complextorch.measures.active_information_storage
   complextorch.measures.transfer_function
   complextorch.measures.inverse_transfer_function
   complextorch.measures.cross_spectral_density
   complextorch.measures.spectral_entropy
   complextorch.measures.state_space_temporal_mvgc
   complextorch.measures.pairwise_spectral_gc
   complextorch.measures.emergence_measures
   complextorch.measures.emergence_from_observations
   complextorch.measures.covariance_amplification
   complextorch.measures.dominant_timescale
   complextorch.measures.stability_margin
   complextorch.measures.gaussian_phiid_mmi
   complextorch.measures.gaussian_phiid_atoms

Synthetic systems and covariance generators
-------------------------------------------

Synthetic constructors define dynamical systems or covariance structures; they
do not simulate observation trajectories. :func:`~complextorch.synthetic_var`
constructs canonical batched VAR(1) systems for controlled sweeps over topology,
spectral radius, innovation correlation, and topology-specific parameters.

.. api-table::

   complextorch.available_synthetic_systems
   complextorch.synthetic_system_parameters
   complextorch.synthetic_transition_matrix
   complextorch.equicorrelated_innovation_covariance
   complextorch.planted_module_projection
   complextorch.synthetic_var
   complextorch.demo_var
   complextorch.random_stable_var
   complextorch.random_correlation_matrix
   complextorch.random_positive_definite_covariance

Trajectory simulation
---------------------

Trajectory generation is kept separate from synthetic model construction.
:func:`~complextorch.simulate_var` simulates independent stationary Gaussian VAR
trajectories from supplied coefficients and innovation covariance.

.. api-table::

   complextorch.automatic_burnin
   complextorch.simulate_var

Spectra and multiscale utilities
-------------------------------

.. api-table::

   complextorch.innovations_spectral_density
   complextorch.integrate_spectral_rate
   complextorch.downsample_innovations_state_space
   complextorch.varma_to_innovations_state_space

Adapters
--------

.. api-table::

   complextorch.from_complexbox_timeseries
   complextorch.to_complexbox_timeseries
   complextorch.from_complexbox_var
   complextorch.to_complexbox_var
