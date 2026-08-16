O-information rate
==================

The O-information rate (OIR) extends static O-information from random variables
to stationary random processes. For :math:`N` process groups
:math:`X_1,\ldots,X_N`, :func:`~complextorch.o_information_rate` implements

.. math::

   \dot\Omega(X_1,\ldots,X_N)
   =(N-2)\dot H(X)
   +\sum_{i=1}^{N}\left[
      \dot H(X_i)-\dot H(X_{-i})
   \right].

For Gaussian innovations processes, entropy-rate constants cancel exactly. If
:math:`V`, :math:`V_i`, and :math:`V_{-i}` are the exact innovations
covariances of the full grouped process, group :math:`i`, and the leave-one-out
process, respectively, the implemented temporal identity is

.. math::

   \dot\Omega
   =\frac{1}{2\log b}\left[
      \sum_i\log|V_i|
      -\sum_i\log|V_{-i}|
      +(N-2)\log|V|
   \right].

Channels not listed in ``groups`` are marginalized out exactly through the
canonical innovations reduction. With two groups, OIR is identically zero.
Positive OIR is redundancy-dominated and negative OIR synergy-dominated.

:func:`~complextorch.spectral_o_information_rate` uses exact marginal
spectral-density matrices:

.. math::

   \omega(f)
   =\frac{1}{2\log b}\left[
      \sum_i\log|S_i(f)|
      -\sum_i\log|S_{-i}(f)|
      +(N-2)\log|S(f)|
   \right].

Whole-band spectral integration recovers temporal OIR up to numerical
quadrature error.

For large batches, ``marginalization="spectrum"`` evaluates all required
marginals as submatrices of one full spectral density and therefore avoids
repeated generalized-DARE reductions. ``marginalization="dare"`` preserves the
original reduced-innovations path and remains the default. The two paths are
frequency-resolved identities; only subsequent numerical integration introduces
quadrature error.

O-information gradient / delta O-information rate
-------------------------------------------------

For selected group :math:`j`, :func:`~complextorch.delta_o_information_rate`
implements Faes et al.'s rate form

.. math::

   \Delta\dot\Omega_j
   =(2-N)\dot I(X_j;X_{-j})
   +\sum_{m\ne j}\dot I(X_j;X_{-\{j,m\}}).

This is exactly

.. math::

   \Delta\dot\Omega_j
   =\dot\Omega(X_1,\ldots,X_N)
   -\dot\Omega(X_1,\ldots,X_{j-1},X_{j+1},\ldots,X_N).

The temporal implementation is built from the independently validated Gaussian
MIR primitive rather than by subtracting two OIR evaluations.
:func:`~complextorch.spectral_delta_o_information_rate` uses the spectral MIR
primitive analogously, providing an independent path for the defining identity.

See :doc:`../../measures` for shared scientific and repository references.
