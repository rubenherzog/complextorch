Information rates and directed dependence
=========================================

Multivariate Granger causality
------------------------------

Partition variables into target :math:`X`, source :math:`Y`, and conditioning
set :math:`Z`. :func:`~complextorch.temporal_mvgc` evaluates time-domain
conditional MVGC,

.. math::

   F_{Y\to X\mid Z}
   =\log_b\frac{\det\Sigma^R_{XX}}{\det\Sigma_{XX}},

where :math:`\Sigma_{XX}` is the target innovation covariance in the full
process and :math:`\Sigma^R_{XX}` is the target innovation covariance after
removing source history. In the model-derived state-space path, the reduced
covariance is obtained by exact generalized-DARE marginalization.

The innovations transfer function is

.. math::

   H(z)=I+C(zI-A)^{-1}K.

:func:`~complextorch.spectral_mvgc` follows the Geweke/MVGC state-space
construction and produces :math:`f_{Y\to X\mid Z}(\nu)`. On normalized
one-sided frequency :math:`\nu\in[0,1/2]`,

.. math::

   F_{Y\to X\mid Z}
   =2\int_0^{1/2}f_{Y\to X\mid Z}(\nu)\,d\nu.

This temporal/spectral identity is an important numerical validation check.
When ``conditional=None`` in the model-derived state-space API, ComplexTorch
uses the MVGC/ComplexBox convention of conditioning on all channels outside the
target and source groups. An explicitly empty conditioning set requests the
unconditioned calculation where supported.

Gaussian information rates
--------------------------

For stationary Gaussian blocks :math:`X,Y`, let :math:`V_X`, :math:`V_Y`, and
:math:`V_{XY}` be exact marginal/joint innovations covariances.
:func:`~complextorch.gaussian_mutual_information_rate` evaluates

.. math::

   \dot I(X;Y)
   =\frac12\log_b\frac{|V_X||V_Y|}{|V_{XY}|}.

The corresponding spectral density from
:func:`~complextorch.spectral_gaussian_mutual_information_rate` is

.. math::

   i_{X;Y}(f)
   =\frac12\log_b
   \frac{|S_X(f)||S_Y(f)|}{|S_{XY}(f)|}.

For source :math:`S` and target :math:`T`,
:func:`~complextorch.gaussian_transfer_entropy_rate` evaluates

.. math::

   \dot T_{S\to T}
   =\frac12\log_b\frac{|V_T^R|}{|V_T|},

where :math:`V_T^R` is the exact target-only marginal innovation covariance and
:math:`V_T` is the target block of the joint source-target innovations
covariance. Thus, under the Gaussian convention,

.. math::

   \dot T_{S\to T}=\frac12F_{S\to T}.

For a joint innovations covariance

.. math::

   V=\begin{pmatrix}V_{XX}&V_{XY}\\V_{YX}&V_{YY}\end{pmatrix},

:func:`~complextorch.gaussian_instantaneous_information_rate` evaluates

.. math::

   \dot I_{X\circ Y}
   =\frac12\log_b\frac{|V_{XX}||V_{YY}|}{|V|}.

See :doc:`../measures` for shared scientific and repository references.
