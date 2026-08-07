API reference
=============

This Phase 1 reference is a compact index of the public ComplexTorch API. It is
built from the package objects with Sphinx ``autosummary`` while the more
detailed narrative and per-object API pages are developed in later
documentation phases.

VAR models and representations
------------------------------

.. autosummary::

   ~complextorch.VAR
   ~complextorch.VARParameters
   ~complextorch.VARSystem
   ~complextorch.build_var_system
   ~complextorch.companion_matrix

State-space models
------------------

.. autosummary::

   ~complextorch.StateSpaceModel
   ~complextorch.InnovationsStateSpace
   ~complextorch.N4SID
   ~complextorch.LarimoreStateSpace
   ~complextorch.LinearGaussianEM
   ~complextorch.kalman_filter
   ~complextorch.kalman_smoother

Model selection
---------------

.. autosummary::

   ~complextorch.EpochTimeSeriesSplit
   ~complextorch.VAROrderSelectionIC
   ~complextorch.VAROrderSearchCV
   ~complextorch.StateSpaceOrderSelection
   ~complextorch.StateSpaceOrderSearchCV

Control and Riccati methods
---------------------------

.. autosummary::

   ~complextorch.solve_dare
   ~complextorch.solve_generalized_dare
   ~complextorch.innovations_form
   ~complextorch.var_to_innovations_state_space
   ~complextorch.reduce_state_space
   ~complextorch.reduce_innovations_state_space
   ~complextorch.project_state_space

Dynamical-dependence optimization
---------------------------------

.. autosummary::

   ~complextorch.optimise_dynamical_dependence
   ~complextorch.DDObjective
   ~complextorch.DDOptimizer
   ~complextorch.DDOptimizationResult
   ~complextorch.dynamical_dependence
   ~complextorch.stochastic_interaction

Measures
--------

.. autosummary::

   ~complextorch.temporal_mvgc
   ~complextorch.spectral_mvgc
   ~complextorch.gaussian_mutual_information_rate
   ~complextorch.gaussian_transfer_entropy_rate
   ~complextorch.gaussian_instantaneous_information_rate
   ~complextorch.spectral_gaussian_mutual_information_rate
   ~complextorch.spectral_gaussian_transfer_entropy_rate
   ~complextorch.phiid_from_model
   ~complextorch.compute_all_model_measures
   ~complextorch.residual_whiteness

Simulation, spectra, and multiscale utilities
---------------------------------------------

.. autosummary::

   ~complextorch.simulate_var
   ~complextorch.demo_var
   ~complextorch.random_stable_var
   ~complextorch.innovations_spectral_density
   ~complextorch.integrate_spectral_rate
   ~complextorch.downsample_innovations_state_space
   ~complextorch.varma_to_innovations_state_space
