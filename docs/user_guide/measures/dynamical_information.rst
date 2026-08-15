Dynamical information
=====================

Autocovariances, spectra, and predictive quantities
---------------------------------------------------

For a stationary state-space/companion representation with transition
:math:`A`, stationary state covariance :math:`P`, and observation matrix
:math:`C`, ComplexTorch uses

.. math::

   \Gamma_k=CA^kPC^{\mathsf T}.

For a VAR(:math:`p`), define

.. math::

   A(f)=I-\sum_{k=1}^{p}A_k e^{-2\pi i f k/f_s},

.. math::

   H(f)=A(f)^{-1},

and

.. math::

   S(f)=\frac1{f_s}H(f)\Sigma H(f)^*.

Here :math:`f_s` is sampling frequency and ``*`` denotes conjugate transpose.
The public dynamics layer includes :func:`~complextorch.measures.entropy_rate`,
:func:`~complextorch.measures.predictive_information`,
:func:`~complextorch.measures.active_information_storage`,
:func:`~complextorch.measures.transfer_function`, and
:func:`~complextorch.measures.cross_spectral_density`.

The Gaussian entropy rate of an innovations process with covariance :math:`V`
is

.. math::

   \dot H(X)=\frac12\log_b\left[(2\pi e)^n\det V\right].

For a stable VAR, ComplexTorch's predictive information is

.. math::

   I_{\mathrm{pred}}
   =\frac12\log_b\frac{\det\Gamma_0}{\det\Sigma}.

For variable :math:`i`, active information storage is

.. math::

   \mathrm{AIS}_i
   =I\left(X_t^{(i)};
            X_{t-1}^{(i)},\ldots,X_{t-p}^{(i)}\right).

See :doc:`../measures` for shared scientific and repository references.
