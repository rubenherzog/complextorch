Predictive emergence
====================

For :math:`Y_t=MX_t`, ComplexTorch defines

.. math::

   \Psi=I(Y_t;X^-)-I(Y_t;Y^-),

.. math::

   \Delta=I(Y_t;Y^-)-\sum_jI(Y_t^j;Y_j^-),

.. math::

   \Gamma=I(Y_t;X^-)-\sum_jI(Y_t^j;X^-).

The public entry points :func:`~complextorch.measures.emergence_measures` and
:func:`~complextorch.measures.emergence_from_observations` should be interpreted
separately from SSDI dynamical dependence even though both compare microscopic
and macroscopic predictive structure.

See :doc:`../../measures` for shared scientific and repository references.
