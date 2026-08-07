API reference
=============

This page documents the public ComplexTorch API directly with Sphinx autodoc.
Functions and classes therefore expose a ``[source]`` link next to their
signature through :mod:`sphinx.ext.viewcode`, without an intermediate generated
API page.

VAR models and representations
------------------------------

.. currentmodule:: complextorch

.. autoclass:: VAR

.. autoclass:: VARParameters

.. autoclass:: VARSystem

.. autofunction:: build_var_system

.. autofunction:: companion_matrix

State-space models
------------------

.. autoclass:: StateSpaceModel

.. autoclass:: InnovationsStateSpace

.. autoclass:: N4SID

.. autoclass:: LarimoreStateSpace

.. autoclass:: LinearGaussianEM

.. autofunction:: kalman_filter

.. autofunction:: kalman_smoother

.. autofunction:: innovations_form

.. autofunction:: innovations_transfer_function

.. autofunction:: var_to_innovations_state_space

.. autofunction:: reduce_state_space

.. autofunction:: reduce_innovations_state_space

.. autofunction:: project_state_space

Model selection
---------------

.. autoclass:: EpochTimeSeriesSplit

.. autoclass:: VAROrderSelectionIC

.. autoclass:: VAROrderSearchCV

.. autoclass:: VAROrderScore

.. autoclass:: VAROrderSearchResult

.. autoclass:: VARInformationCriteriaResult

.. autoclass:: StateSpaceOrderSelection

.. autoclass:: StateSpaceOrderSelectionResult

.. autoclass:: StateSpaceOrderSearchCV

.. autoclass:: StateSpaceOrderScore

.. autoclass:: StateSpaceOrderSearchResult

Control and Riccati methods
---------------------------

.. autofunction:: solve_dare

.. autofunction:: solve_generalized_dare

.. autoclass:: ProjectionSearchResult

.. autofunction:: optimise_dynamical_dependence_projection

Dynamical-dependence optimization
---------------------------------

:func:`optimise_dynamical_dependence` uses the canonical staged SSDI workflow
when ``objective=None`` and returns :class:`DDSSDIOptimizationResult`. Explicit
single-stage proxy or spectral optimization returns
:class:`DDOptimizationResult`.

.. autofunction:: optimise_dynamical_dependence

.. autodata:: DDObjective

.. autodata:: DDOptimizer

.. autoclass:: DDOptimizationResult

.. autoclass:: DDSSDIOptimizationResult

.. autoclass:: DDGradientSearchResult

.. autofunction:: dynamical_dependence

.. autofunction:: stochastic_interaction

.. autofunction:: innovations_proxy_sequence

.. autofunction:: optimise_dynamical_dependence_proxy

.. autofunction:: optimise_dynamical_dependence_spectral

.. autofunction:: orthonormalise_projection

.. autofunction:: proxy_dynamical_dependence

.. autofunction:: proxy_dynamical_dependence_gradient

.. autofunction:: spectral_dynamical_dependence

.. autofunction:: spectral_dynamical_dependence_gradient

Model-derived measures
----------------------

.. autoclass:: ModelMeasureConfig

.. autoclass:: ModelMeasureContext

.. autofunction:: build_measure_context

.. autofunction:: compute_all_model_measures

.. autofunction:: model_autocovariances

.. autofunction:: temporal_mvgc

.. autofunction:: spectral_mvgc

.. autofunction:: gaussian_mutual_information_rate

.. autofunction:: gaussian_transfer_entropy_rate

.. autofunction:: gaussian_instantaneous_information_rate

.. autofunction:: spectral_gaussian_mutual_information_rate

.. autofunction:: spectral_gaussian_transfer_entropy_rate

.. autofunction:: o_information_rate

.. autofunction:: spectral_o_information_rate

.. autofunction:: delta_o_information_rate

.. autofunction:: spectral_delta_o_information_rate

.. autofunction:: partial_information_rate_decomposition

.. autofunction:: spectral_partial_information_rate_decomposition

.. autoclass:: PIRDResult

.. autoclass:: SpectralPIRDResult

.. autofunction:: partial_granger_causality_decomposition

.. autofunction:: spectral_partial_granger_causality_decomposition

.. autoclass:: PDGCResult

.. autoclass:: SpectralPDGCResult

.. autofunction:: hop_analysis

.. autofunction:: spectral_hop_analysis

.. autoclass:: HOPResult

.. autoclass:: SpectralHOPResult

.. autofunction:: phiid_from_model

.. autofunction:: phiid_redundancy_from_model

.. autoclass:: WhitenessResult

.. autofunction:: residual_whiteness

.. autofunction:: consistency

.. autofunction:: mvgc_pvalue

.. autofunction:: significance

Gaussian covariance primitives
------------------------------

These public primitives live in :mod:`complextorch.measures` and are useful
when the covariance matrix itself is the scientific input.

.. currentmodule:: complextorch.measures

.. autofunction:: conditional_covariance

.. autofunction:: gaussian_conditional_mutual_information

.. autofunction:: total_correlation

.. autofunction:: dual_total_correlation

.. autofunction:: o_information

.. autofunction:: s_information

.. autofunction:: local_gaussian_mutual_information

Dynamics and additional measure primitives
------------------------------------------

.. autofunction:: entropy_rate

.. autofunction:: predictive_information

.. autofunction:: active_information_storage

.. autofunction:: transfer_function

.. autofunction:: inverse_transfer_function

.. autofunction:: cross_spectral_density

.. autofunction:: spectral_entropy

.. autofunction:: state_space_temporal_mvgc

.. autofunction:: pairwise_spectral_gc

.. autofunction:: emergence_measures

.. autofunction:: emergence_from_observations

.. autofunction:: covariance_amplification

.. autofunction:: dominant_timescale

.. autofunction:: stability_margin

.. autofunction:: gaussian_phiid_mmi

.. autofunction:: gaussian_phiid_atoms

Simulation, spectra, and multiscale utilities
---------------------------------------------

.. currentmodule:: complextorch

.. autofunction:: automatic_burnin

.. autofunction:: simulate_var

.. autofunction:: demo_var

.. autofunction:: random_stable_var

.. autofunction:: random_correlation_matrix

.. autofunction:: random_positive_definite_covariance

.. autofunction:: innovations_spectral_density

.. autofunction:: integrate_spectral_rate

.. autofunction:: downsample_innovations_state_space

.. autofunction:: varma_to_innovations_state_space

Adapters
--------

.. autofunction:: from_complexbox_timeseries

.. autofunction:: to_complexbox_timeseries

.. autofunction:: from_complexbox_var

.. autofunction:: to_complexbox_var
