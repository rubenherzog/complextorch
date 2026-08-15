PhiID
=====

For two variables and a positive lag :math:`\tau`,
:func:`~complextorch.phiid_from_model` constructs the joint Gaussian covariance
of

.. math::

   (X_t^1,X_t^2,X_{t+\tau}^1,X_{t+\tau}^2)

from model autocovariances. The redundancy prescription can be evaluated through
:func:`~complextorch.phiid_redundancy_from_model`; see
:doc:`../../phiid_redundancy` for the currently supported Gaussian backends and
their scientific conventions.

The bivariate PhiID lattice uses source and target antichains drawn from

.. math::

   \{\mathrm{red},\mathrm{unq0},\mathrm{unq1},\mathrm{syn}\},

producing 16 source-to-target atoms. Möbius inversion yields the atoms and
satisfies

.. math::

   \sum_{\alpha,\beta}I_{\partial}^{\alpha\to\beta}
   =I\left((X_t^1,X_t^2);(X_{t+\tau}^1,X_{t+\tau}^2)\right).

See :doc:`../../measures` for shared scientific and repository references.
