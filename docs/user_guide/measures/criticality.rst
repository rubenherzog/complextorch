Criticality
===========

Criticality diagnostics
-----------------------

For transition spectral radius :math:`\rho`, ComplexTorch exposes

.. math::

   \text{stability margin}=1-\rho,

.. math::

   \tau=-\frac{\Delta t}{\log\rho},

and, for VAR systems,

.. math::

   A_{\mathrm{cov}}
   =\frac{\operatorname{tr}\Gamma_0}{\operatorname{tr}\Sigma}.

The corresponding public diagnostics are
:func:`~complextorch.measures.stability_margin`,
:func:`~complextorch.measures.dominant_timescale`, and
:func:`~complextorch.measures.covariance_amplification`. These are linear-system
diagnostics and should not alone be interpreted as evidence of a physical phase
transition.

See :doc:`../measures` for shared scientific and repository references.
