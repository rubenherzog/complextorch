Dynamical dependence and staged SSDI
====================================

Scientific quantity
-------------------

Dynamical dependence (DD) quantifies how much predictive information in a
microscopic stationary Gaussian process is lost after projection onto a
lower-dimensional macroprocess. Let

.. math::

   Y_t=LX_t,
   \qquad L\in\mathbb R^{m\times n},

and let :math:`\Sigma` be the microscopic innovations covariance. If
:math:`\Sigma_R` is the exact innovations covariance of the projected process,
:func:`~complextorch.dynamical_dependence` evaluates

.. math::

   F(X\to Y)
   =\log_b\frac{|\Sigma_R|}{|L\Sigma L^{\mathsf T}|}.

Under the Gaussian Shannon convention the corresponding transfer-entropy rate
is

.. math::

   T(X\to Y)=\frac12F(X\to Y).

ComplexTorch's DD functions return :math:`F`, not :math:`F/2`. The default log
base of the scalar DD measure is 2. Use natural logarithms when direct
natural-log SSDI/ComplexBox parity is required.

Evaluation versus optimization
------------------------------

The public API separates the **scientific DD quantity** from the **search over
macro subspaces**. :func:`~complextorch.dynamical_dependence` evaluates DD for a
supplied projection and belongs to the evaluation layer implemented in
``dd.py``. The optimization objectives, gradients, numerical step policies,
and canonical staged SSDI search are implemented in ``dd_optimization.py`` and
exposed through :func:`~complextorch.optimise_dynamical_dependence` and its
lower-level helpers.

This architectural separation does not define a second DD measure and does not
change the SSDI scientific workflow. It keeps evaluation usable independently
from optimization and prevents adaptive and Armijo search machinery from being
duplicated across intermediate modules.

Subspaces, not bases
--------------------

ComplexTorch stores macro projections as row bases with shape ``(m, n)`` or
``(runs, m, n)``. Optimized representatives are row-orthonormal,

.. math::

   LL^{\mathsf T}=I_m.

The scientific object is the Grassmann subspace spanned by the rows. Replacing
:math:`L` by :math:`UL` for any orthogonal :math:`U\in\mathbb R^{m\times m}`
does not change that subspace. Raw matrix entries should therefore not be used
to compare optimizer endpoints.

Canonical public optimizer
--------------------------

:func:`~complextorch.optimise_dynamical_dependence` is the public dispatcher.
The default ``objective=None`` no longer denotes a single objective. It runs the
validated staged SSDI search procedure and returns
:class:`~complextorch.DDSSDIOptimizationResult`.

The default call therefore has the scientific structure

.. math::

   \text{many proxy restarts}
   \longrightarrow
   \text{Grassmann clustering}
   \longrightarrow
   \text{spectral refinement}.

Explicit ``objective="proxy"`` or ``objective="spectral"`` requests bypass the
staged workflow and preserve the single-stage research/backward-compatibility
API, returning :class:`~complextorch.DDOptimizationResult`.

Stage 1: proxy pre-optimization
-------------------------------

For an innovations-form state-space system, the SSDI proxy sequence is

.. math::

   Q_k=CA^{k-1}K.

For a row projection :math:`L`, the identity-innovation proxy objective is

.. math::

   D_x(L)
   =\sum_k\|LQ_k\|_F^2
    -\|LQ_kL^{\mathsf T}\|_F^2.

The low-level implementation is
:func:`~complextorch.proxy_dynamical_dependence`, with sequence construction by
:func:`~complextorch.innovations_proxy_sequence` where that representation is
used.

For a general innovations covariance :math:`V=BB^{\mathsf T}`, ComplexTorch
whitens observation coordinates before optimization and maps the optimized
subspace back by triangular solves. The scientific process is unchanged; no
explicit covariance inverse is formed.

For :class:`~complextorch.VARSystem` in the staged SSDI workflow, the proxy
stage follows the MATLAB SSDI ``cak2ddx`` convention more directly. The
innovation-whitened VAR coefficients are

.. math::

   A_k^{(w)}=B^{-1}A_kB,

and the physical row projection maps to whitened coordinates as :math:`LB`.
This avoids replacing a finite-order VAR proxy by an unnecessarily expanded
state-space proxy sequence.

When ``initial_projection`` is omitted, ``output_dimension`` must be supplied.
The public staged API then generates ``preoptimization_runs=100`` independent
random row-orthonormal restart subspaces by default. ``random_seed=0`` is the
reproducible default; ``None`` requests nondeterministic sampling.

Stage 2: SSDI Grassmann clustering
----------------------------------

After proxy optimization, runs are sorted by final proxy objective. Pairwise
subspace distance follows SSDI/ComplexBox ``gmetrics`` semantics. For two
row-orthonormal representatives :math:`L_a,L_b`, let
:math:`\sigma_{\min}` be the smallest singular value of
:math:`L_aL_b^{\mathsf T}`. The maximum principal angle is

.. math::

   \theta_{\max}=\arccos\sigma_{\min},

and the normalized distance is

.. math::

   d(L_a,L_b)
   =\frac{\theta_{\max}}{\pi/2}
   =\frac{2}{\pi}\arccos\sigma_{\min}(L_aL_b^{\mathsf T}).

The result lies in :math:`[0,1]`. The diagonal is zero and the matrix is
symmetric.

The default ``cluster_tolerance=0.01`` implements the strict SSDI ``Lcluster``
criterion. Iterating through proxy minima in ascending objective order, the
first still-available run becomes a representative and all still-available
runs with distance strictly below the tolerance join its cluster. Thus each
cluster is represented by its lowest-objective encountered member.

Stage 3: spectral refinement
----------------------------

One proxy representative per cluster initializes full spectral-DD
optimization. The low-level identity-innovation spectral objective is evaluated
by :func:`~complextorch.spectral_dynamical_dependence`. For transfer matrices
:math:`H(f_j)`, ComplexTorch evaluates the projected log-determinant spectrum
with the same one-sided trapezoidal normalization used by SSDI/ComplexBox.

If ``frequencies`` is omitted in staged mode, the default grid contains
``frequency_points=513`` equally spaced points over normalized one-sided
frequency :math:`[0,1/2]`. The default ceilings are
``preoptimization_max_iterations=10000`` and
``spectral_max_iterations=10000``.

Numerical step policies
-----------------------

``optimizer="adaptive"`` is the default numerical step policy. It implements
the ComplexBox-compatible adaptive search behavior, and staged mode uses the
scientifically recommended variant 1 for proxy pre-optimization and spectral
refinement unless a stage-specific variant is explicitly supplied.

``optimizer="armijo"`` replaces the numerical update rule with native batched
Riemannian Armijo backtracking. It uses the **same** proxy objective, Grassmann
clustering rule, spectral objective, and staged scientific workflow; therefore
``adaptive`` and ``armijo`` should not be interpreted as different definitions
of dynamical dependence.

The historical names ``"complexbox"`` and ``"riemannian_armijo"`` are accepted
as compatibility aliases for ``"adaptive"`` and ``"armijo"`` respectively.
New analyses and documentation should use the current names.

Stage-specific numerical options are passed with ``preoptimizer_options`` and
``spectral_optimizer_options``. The legacy ``optimizer_options`` and
``max_iterations`` arguments are intentionally rejected in staged mode because
the two optimization stages have separate controls.

Result contract
---------------

The staged result :class:`~complextorch.DDSSDIOptimizationResult` contains:

``preoptimization``
   A :class:`~complextorch.DDOptimizationResult` for all proxy restarts.

``cluster_representative_indices``
   Indices of the retained proxy representatives in sorted proxy-result order.

``cluster_sizes``
   Number of proxy minima assigned to each cluster.

``cluster_distances``
   Complete pairwise normalized Grassmann-distance matrix for the proxy
   endpoints.

``spectral``
   A :class:`~complextorch.DDOptimizationResult` containing the final spectral
   refinements, one run per retained cluster representative.

``frequencies``
   Frequency grid used by the spectral stage.

The convenience ``objective`` and ``projection`` properties are aliases for
``spectral.objective`` and ``spectral.projection``. They therefore describe the
final full spectral-DD solutions rather than proxy pre-optimization.

The common :class:`~complextorch.DDOptimizationResult` stores sorted objective
values, physical-coordinate projections, iteration counts, convergence flags
and raw codes, final step sizes, objective/gradient/backtracking evaluation
counts, optional normalized history, numerical policy name, and objective name.
Compatibility aliases are normalized to the current ``adaptive``/``armijo``
names before the optimization result is produced.

Single-system scope
-------------------

The current DD optimization machinery accepts one microscopic dynamical system
at a time. A singleton batch representation is accepted, but optimization over
a batch of distinct microscopic systems is outside the current parity scope.
The ``runs`` dimension belongs to optimizer restart subspaces; it must not be
confused with the independent-trajectory batch dimension used by model fitting.

Recommended reporting
---------------------

For a reproducible staged SSDI analysis, report at least:

- repository commit and model representation;
- microscopic dimension :math:`n` and macro dimension :math:`m`;
- numerical step policy (``adaptive`` or ``armijo``);
- number and source of initial restart subspaces;
- random seed when restarts are generated;
- proxy lag configuration;
- cluster tolerance;
- complete frequency grid or ``frequency_points``;
- proxy and spectral iteration ceilings;
- stage-specific optimizer options;
- number and sizes of retained clusters;
- convergence codes and objective values for both stages;
- final spectral-DD objectives;
- basis-invariant distances when comparing optimized subspaces.

For optimizer studies, wall time, objective/gradient evaluation counts, local
minimum structure, and principal-angle distances should be reported separately.

Single-stage calls
------------------

Explicit single-stage optimization remains available for method development,
parity tests, and diagnostics:

``objective="proxy"``
   Requires ``initial_projection``. ``lags`` is optional. ``frequencies`` is
   invalid.

``objective="spectral"``
   Requires both ``initial_projection`` and an explicit ``frequencies`` grid.
   ``lags`` is invalid.

These calls return :class:`~complextorch.DDOptimizationResult` and should not be
reported as the default SSDI workflow.

References
----------

- Barnett, L. and Seth, A. K. (2023). Dynamical independence: Discovering
  emergent macroscopic processes in complex dynamical systems. *Physical
  Review E*, 108, 014304.
- SSDI/ComplexBox reference workflow: proxy multi-start optimization,
  ``Lcluster``, and full spectral optimization.

Repository references
---------------------

- ``src/complextorch/control.py``
- ``src/complextorch/dd.py``
- ``src/complextorch/dd_optimization.py``
- ``tests/test_dd_ssdi.py``
- ``tests/test_ssdi_validation.py``
