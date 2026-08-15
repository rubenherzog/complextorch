Gaussian information
====================

Gaussian entropy and mutual information
---------------------------------------

For :math:`X\sim\mathcal N(\mu,\Sigma)`, :math:`X\in\mathbb R^d`,

.. math::

   H(X)=\frac12\log_b\left[(2\pi e)^d\det\Sigma\right].

For jointly Gaussian :math:`X,Y`,

.. math::

   I(X;Y)
   =\frac12\log_b
   \frac{\det\Sigma_X\det\Sigma_Y}{\det\Sigma_{XY}}.

For jointly Gaussian :math:`X,Y,Z`,

.. math::

   I(X;Y\mid Z)
   =\frac12\log_b
   \frac{\det\Sigma_{XZ}\det\Sigma_{YZ}}
        {\det\Sigma_Z\det\Sigma_{XYZ}}.

Log determinants are evaluated through SPD-aware numerical routines.

Static high-order information
-----------------------------

For :math:`X=(X_1,\ldots,X_N)`, total correlation is

.. math::

   \mathrm{TC}(X)=\sum_{i=1}^NH(X_i)-H(X),

and dual total correlation is

.. math::

   \mathrm{DTC}(X)
   =\sum_{i=1}^NH(X_{-i})-(N-1)H(X),

where :math:`X_{-i}` denotes all variables except :math:`X_i`.

The covariance-level implementations are
:func:`~complextorch.measures.total_correlation` and
:func:`~complextorch.measures.dual_total_correlation`. ComplexTorch defines
O-information through :func:`~complextorch.measures.o_information` as

.. math::

   \Omega(X)=\mathrm{TC}(X)-\mathrm{DTC}(X),

and S-information through :func:`~complextorch.measures.s_information` as

.. math::

   S(X)=\mathrm{TC}(X)+\mathrm{DTC}(X).

Positive :math:`\Omega` indicates redundancy-dominated high-order dependence;
negative :math:`\Omega` indicates synergy-dominated dependence.

See :doc:`../measures` for shared scientific and repository references.
