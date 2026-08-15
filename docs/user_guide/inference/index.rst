Inference and reference normalisation
=====================================

ComplexTorch keeps two distinct post-fit questions separate:

- :doc:`confidence_intervals` quantifies sampling uncertainty of model-derived
  measures using resampled VAR fits;
- :doc:`numit` compares PID atoms against a constrained null ensemble for
  cross-system normalisation.

These procedures answer different scientific questions. Confidence intervals
estimate uncertainty around a measure for an observed time series. NuMIT is a
null-reference normalisation and is not a bootstrap or confidence-interval
method.

.. toctree::
   :maxdepth: 2

   confidence_intervals
   numit
