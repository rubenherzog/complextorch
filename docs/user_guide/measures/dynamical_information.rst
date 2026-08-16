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

Full-past CMem1 and scalable marginal entropy rates
---------------------------------------------------

For full-past component histories, ComplexTorch exposes
:func:`~complextorch.cmem1_full_past`:

.. math::

   CMem_1
   =[H(X_t)-h(X)]
   -\sum_i [H(X_t^i)-h(X^i)].

The marginal rates :math:`h(X^i)` are rates of the exact marginal processes;
they are not obtained by truncating a marginal VAR at the microscopic VAR
order. By default ``marginal_method="dare"`` uses exact generalized-DARE
marginal innovations. For large feature-extraction workloads,
``marginal_method="spectral"`` uses one full spectral density and integrates
its scalar marginal spectra. The latter is a numerical quadrature of the same
full-past quantity, so the frequency grid and integration convention should be
reported.

The same choice is available directly through
:func:`~complextorch.marginal_entropy_rate`. The DARE path remains the default;
the spectral path is an opt-in efficiency extension.

Scalable singleton temporal MVGC
--------------------------------

:func:`~complextorch.pairwise_temporal_mvgc` computes the complete conditional
singleton MVGC matrix while reusing one reduced innovations model per source.
This is algebraically identical to evaluating each ordered source--target pair
separately, but reduces the number of generalized-DARE marginalizations from
quadratic to linear in the number of variables.
:func:`~complextorch.maximum_temporal_mvgc` returns the corresponding maximum
without changing the underlying conditional-MVGC definition.

Scalable pairwise mutual-information-rate features
--------------------------------------------------

For high-throughput feature extraction,
:func:`~complextorch.pairwise_gaussian_mutual_information_rate` and
:func:`~complextorch.mean_pairwise_gaussian_mutual_information_rate` expose the
full singleton-pair MIR matrix and its unordered-pair mean. The default
``method="dare"`` retains the exact reduced-innovations calculation. The opt-in
``method="spectral"`` evaluates one full model spectrum, extracts all pairwise
spectral submatrices in batch, and numerically integrates their MIR densities.
This avoids repeated generalized-DARE reductions while preserving the same
scientific quantity up to the explicitly controlled spectral quadrature error.

Reusable spectral measure context
---------------------------------

When several spectral features are evaluated on the same model and frequency
grid, :func:`~complextorch.build_spectral_measure_context` computes the full
observation spectrum once and stores it with its frequency metadata in
:class:`~complextorch.SpectralMeasureContext`. The scalable spectral backends
for marginal entropy rates/CMem1, pairwise MIR, OIR, and PIRD extrema accept
this context through ``spectral_context=``. Reusing the context changes only
computation reuse; all measure definitions, frequency grids, and quadrature
conventions remain explicit and unchanged.

A context is tied scientifically to the model from which it was built. It
should therefore only be reused for measures of that same model, on the same
frequency grid and sampling frequency. ComplexTorch validates the spectral
metadata and observation dimension; callers remain responsible for preserving
model identity when caching contexts across a larger analysis.
