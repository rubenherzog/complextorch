User guide
==========

This guide documents the mathematical and scientific conventions implemented by
ComplexTorch. It complements the :doc:`../api` reference: the API pages describe
signatures and return types, while these pages explain models, equations, tensor
semantics, estimation assumptions, numerical choices, and interpretation.

The guide is grounded in the implementation on the repository ``main`` branch.
When external software is mentioned, it is a parity/reference implementation;
the cited scientific literature remains the primary mathematical authority.

.. toctree::
   :maxdepth: 2

   conventions
   var
   state_space
   selection
   control
   dynamical_dependence
   measures
   phiid_redundancy
   numerics_reproducibility
