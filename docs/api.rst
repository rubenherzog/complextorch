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
   complextorch.pairwise_temporal_mvgc
   complextorch.maximum_temporal_mvgc
   complextorch.spectral_mvgc
   complextorch.gaussian_mutual_information_rate
   complextorch.pairwise_gaussian_mutual_information_rate
   complextorch.mean_pairwise_gaussian_mutual_information_rate
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
   complextorch.PIRDExtremaResult
   complextorch.pird_extrema
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

Fit and recovery diagnostics
----------------------------

In-sample and out-of-sample evaluation are exposed through the same public
:func:`~complextorch.fit_diagnostics` entry point and share the
:func:`~complextorch.innovation_diagnostics` statistical core. Ground-truth
state-space recovery is a separate operation for simulations and controlled
benchmarks and uses gauge-aware model invariants rather than raw matrix-entry
errors.

.. api-table::

   complextorch.FitDiagnostics
   complextorch.fit_diagnostics
   complextorch.innovation_diagnostics
   complextorch.StateSpaceRecoveryDiagnostics
   complextorch.state_space_recovery_diagnostics

Inference, confidence intervals, and NuMIT
------------------------------------------

Sampling uncertainty and constrained null-reference normalisation are separate
operations. :func:`~complextorch.measure_confidence_intervals` accepts fixed-
complexity VAR, innovations-form, or general state-space estimators and evaluates
compatible model-derived measures on one shared residual-bootstrap or Gaussian-
parametric ensemble. State-space surrogate generation is normalized through the
canonical innovations representation. :func:`~complextorch.numit_pid_var`
instead compares Gaussian VAR PID atoms with an otherwise-random ensemble
matched to the observed total past--future mutual information. NuMIT is not a
bootstrap confidence-interval method.

.. api-table::

   complextorch.measure_confidence_intervals
   complextorch.InferenceMeasureConfig
   complextorch.ConfidenceIntervalResult
   complextorch.MeasureInterval
   complextorch.ResamplingMethod
   complextorch.numit_pid_var
   complextorch.var_total_mutual_information
   complextorch.NuMITPIDResult

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
