Partial information rate decomposition (PIRD)
=============================================

:func:`~complextorch.partial_information_rate_decomposition` and
:func:`~complextorch.spectral_partial_information_rate_decomposition` implement
Gaussian partial information rate decomposition for exactly two or three source
groups and one disjoint target group, following the Faes/HOP minimum-MIR
convention.

Let the source groups be :math:`X_1,\ldots,X_M`, with :math:`M\in\{2,3\}`, and
let :math:`Y` be the target process. For every non-empty source subset
:math:`A\subseteq\{1,\ldots,M\}`, first compute the spectral Gaussian mutual
information rate

.. math::

   i_A(f)=i(X_A;Y;f).

For a Williams--Beer redundancy-lattice antichain :math:`\alpha`, the spectral
redundancy function is the frequency-wise minimum

.. math::

   I_{\cap}(\alpha;f)
   =\min_{A\in\alpha}i_A(f).

Partial information spectra are then obtained by Möbius inversion on the PID
lattice. If :math:`I_{\partial}(\alpha;f)` denotes the atom associated with
antichain :math:`\alpha`,

.. math::

   I_{\cap}(\alpha;f)
   =\sum_{\beta\preceq\alpha}I_{\partial}(\beta;f),

or equivalently

.. math::

   I_{\partial}(\alpha;f)
   =I_{\cap}(\alpha;f)
   -\sum_{\beta\prec\alpha}I_{\partial}(\beta;f).

The crucial convention is that the minimum is applied **frequency by
frequency**. Integrated temporal atoms are therefore

.. math::

   \dot I_{\partial}(\alpha)
   =\int I_{\partial}(\alpha;f)\,df,

rather than a new decomposition obtained by first integrating each subset MIR
and then taking a minimum.

The public spectral result :class:`~complextorch.SpectralPIRDResult` exposes
source-subset MIR spectra, redundancy functions, all Möbius-inverted atoms,
unique spectra per source, total redundant and synergistic spectra, and

.. math::

   \Delta=\mathrm{redundant}-\mathrm{synergistic}.

For two sources, the coarse-grained atoms are unique source 1, unique source 2,
redundancy, and synergy. For three sources, the validated Faes/HOP coarse
graining combines the 18-node Williams--Beer lattice into three unique
components plus total redundant and synergistic components. The integrated
result is represented by :class:`~complextorch.PIRDResult`.

The ``half_open=True`` option follows the Faes/HOP half-open frequency-grid
convention and arithmetic-mean integration implemented by
:func:`~complextorch.integrate_spectral_rate`; otherwise endpoint-inclusive
trapezoidal integration is used.

PIRD reuses the shared generalized-DARE reduction, spectral density, MIR,
integration, PID lattice, and Möbius inversion primitives rather than
implementing parallel numerical machinery.

See :doc:`../../measures` for shared scientific and repository references.

Scalable extrema for feature extraction
----------------------------------------

:func:`~complextorch.pird_extrema` is a feature-extraction helper for the
common singleton case used in large-system fingerprinting. It evaluates every
valid pair of singleton sources against every distinct singleton target, then
returns the maximum integrated PIRD synergy and redundancy together with the
attaining ``(source0, source1, target)`` indices. The full spectral density is
computed once and all candidate submatrices are evaluated in Torch batches.

This does not replace the general PIRD API: use
:func:`~complextorch.partial_information_rate_decomposition` when the full atom
structure, grouped variables, or three-source decomposition is required.
