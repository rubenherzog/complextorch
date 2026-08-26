Mechanistic representation and prescribed dynamical design
==========================================================

ComplexTorch separates **process mechanics**, **information capabilities**, and
**design objectives**.  The purpose of this layer is not to assign one graph
statistic to one information measure.  It provides process-level coordinates
and differential tools for asking which dynamical capabilities can be changed,
held fixed, or jointly prescribed by modifying a model architecture.

The implementation is Torch-first and batch preserving.  It does not simulate
trajectories or refit observations: all calculations act directly on canonical
``VARSystem``, ``StateSpaceModel``, or ``InnovationsStateSpace`` objects and on
user-supplied design parameters.

Process-level modal coordinates
-------------------------------

For the innovations representation

.. math::

   x_{t+1}=Ax_t+K\varepsilon_t,\qquad
   y_t=Cx_t+\varepsilon_t,\qquad
   \varepsilon_t\sim\mathcal N(0,V),

ComplexTorch uses

.. math::

   H(z)=I+C(zI-A)^{-1}K.

If ``A`` is diagonalizable with simple poles,

.. math::

   H(z)=I+\sum_j\frac{R_j}{z-\lambda_j},

where

.. math::

   R_j=C P_j K,
   \qquad
   P_j=r_j l_j^*,
   \qquad
   l_j^*r_j=1.

:func:`~complextorch.modal_decomposition` returns ``lambda_j``, ``R_j``, the
largest singular value ``s_j`` of each residue, normalized residues, and the
condition number of the eigenvector basis.  Nonzero-residue pole--residue
pairs are invariant to latent-state similarity transformations.  Distinct
nonminimal state modes are retained with an ``active=False`` mask rather than
being interpreted as observable transfer modes.  Raw state-space matrix entries
are not process invariants.

The decomposition intentionally rejects repeated or numerically unresolved
poles.  Individual simple-mode residues are not unique at repeated poles, and
near-defective eigenvector bases can make them numerically sensitive.  The
returned ``eigenvector_condition`` exposes that sensitivity instead of hiding
it with regularization.

For a fully observed VAR(1), represented in innovations form as

.. math::

   A_{ss}=A_G,\qquad C=A_G,\qquad K=I,

one obtains the exact architecture--mechanism bridge

.. math::

   R_j=A_G P_j=\lambda_jP_j,

and therefore

.. math::

   s_j=|\lambda_j|\,\|P_j\|_2.

For a normal matrix, ``||P_j||_2=1`` and residue strength is locked to pole
magnitude.  For a non-normal matrix, spectral projectors can have norm greater
than one, so responsiveness can increase without moving the poles.  This is a
property of the linear system, not a separate graph statistic.

Master covariance identity
--------------------------

Let

.. math::

   M_h=CA^hK=\sum_j R_j\lambda_j^h.

The stationary observation covariance is

.. math::

   \boxed{
   \Gamma_0
   =V+\sum_{j,k}
   \frac{R_j V R_k^*}{1-\lambda_j\bar\lambda_k}
   }.

:func:`~complextorch.modal_observation_covariance` evaluates this identity
directly.  It is validated against the canonical Lyapunov/autocovariance
backbone rather than implemented as a second covariance definition.

This factorization motivates three useful mechanistic readings:

``Stability``
   The pole locations ``lambda_j`` determine persistence, oscillatory phase,
   and temporal separation of modes.

``Responsiveness``
   Residue magnitudes ``s_j`` determine how strongly modes couple innovations
   to observations.

``Coordination``
   The normalized residue geometry determines how responses are distributed,
   aligned, directed, and able to interfere across modes.

These labels are scientific interpretations of the complete pole--residue
object, not claims that an arbitrary process admits three sufficient scalar
coordinates.

Innovation context and scale
----------------------------

The innovation covariance is separated into a global scale and a normalized
geometry.  ``ModalDecomposition.normalized_innovation_covariance`` returns

.. math::

   \bar V=\frac{V}{\operatorname{tr}V}.

For normalized Gaussian information capabilities derived from the same linear
process, replacing ``V`` by ``beta V`` multiplies all stationary covariances by
``beta`` but cancels from covariance ratios.  Thus predictive information,
full-past collective memory, and dynamical dependence are invariant to this
global innovation scale.  Anisotropy and orientation of ``bar V`` do not, in
general, cancel and remain genuine context variables.

This scale invariance is an exact property under the stated linear-Gaussian
assumptions.  It should not be generalized to unnormalized quantities such as
absolute variance or Gaussian differential entropy.

Exact information-measure relations
-----------------------------------

Several quantities exposed by ComplexTorch are related analytically and should
not be interpreted as independent coordinates merely because they have
separate API functions.

For full-past collective memory,

.. math::

   \boxed{
   CMem_{1,\infty}
   =PI-\sum_i AIS_{i,\infty}
   =TC_{rate}-TC_0
   }.

For a fixed full-row-rank macro projection ``L``, let ``V_L^R`` be the exact
innovations covariance of the projected macroprocess.  Dynamical dependence is

.. math::

   DD(L)=\log_b\frac{\det V_L^R}{\det(LVL^\top)}.

:func:`~complextorch.project_innovations_state_space` exposes the exact
projected innovations representation used by DD.  If

.. math::

   \Gamma_L=L\Gamma_0L^\top,

then the macro predictive information satisfies

.. math::

   PI(L)=\frac12\log_b\frac{\det\Gamma_L}{\det V_L^R},

and therefore

.. math::

   \boxed{
   2PI(L)+DD(L)
   =\log_b\frac{\det\Gamma_L}{\det(LVL^\top)}
   }.

This is an exact fixed-``L`` identity.  It is **not** a universal functional
relationship between microscopic ``CMem`` and ``DD``.  DD depends explicitly
on the chosen macro projection and on the reduced spectral factor.

Differential accessibility and degeneracy
-----------------------------------------

Let ``theta`` denote arbitrary continuous design parameters and let

.. math::

   D(\theta)\in\mathbb R^m

be a vector of dynamical capabilities.  The local design map is

.. math::

   J_{D\leftarrow\theta}=\frac{\partial D}{\partial\theta}.

:func:`~complextorch.finite_difference_jacobian` computes this Jacobian with a
central difference while evaluating all plus/minus perturbations in one batched
call.  This is useful when the capability function is analytical but not
conveniently differentiable end to end.

The local dimension accessible in capability space is the rank of ``J``.  For
fixed target capabilities ``D_A``, the neutral tangent space is

.. math::

   T_\theta\mathcal E_A
   =\operatorname{Null}(J_A),

where

.. math::

   \mathcal E_A(d^*)
   =\{\theta:D_A(\theta)=d^*\}.

:func:`~complextorch.neutral_projector` returns the orthogonal projector onto
this nullspace.  A fixed-shape projector is used instead of a padded nullspace
basis so that batches whose numerical ranks differ remain representable.

If ``D_B`` are untargeted capabilities, their first-order freedom under fixed
``D_A`` is represented by

.. math::

   \boxed{J_B P_{N_A}},

returned by :func:`~complextorch.capability_mobility`.  Nonzero singular values
of this operator quantify **functional sloppiness**: capability combinations
that can move while the prescribed panel is locally unchanged.

Local nullity and global uniqueness are different questions.  A zero-dimensional
local nullspace does not rule out disconnected architectures elsewhere in
parameter space with the same capability vector.

Level-set correction
--------------------

:func:`~complextorch.project_to_capability_level_set` locally corrects a design
toward

.. math::

   D_A(\theta)=d^*.

With residual ``r=D_A(theta)-d*``, it uses the damped minimum-norm step

.. math::

   \Delta\theta
   =J^\top(JJ^\top+\lambda I)^{-1}r.

The solve is batched and no explicit matrix inverse is formed.  Backtracking
candidates for all active designs are also evaluated in a common batch.  An
optional ``validity_function`` can reject candidates that violate external
constraints such as stationarity or parameter bounds.

This is a local projection/correction primitive.  Failure to converge does not
prove that the requested capability vector is globally infeasible.

Prescribed-capability optimization
----------------------------------

:func:`~complextorch.optimise_prescribed_capabilities` implements a minimal
Torch-native multistart design loop.  The leading dimension contains independent
starts, which are optimized simultaneously.  A user supplies

* a differentiable batched capability function,
* one common target or one target per run,
* an optional design objective, and
* an optional additional soft penalty.

The optimization stage minimizes

.. math::

   \lambda_D\|D_A(\theta)-d^*\|_2^2
   +C(\theta,D)
   +R(\theta,D)

with Adam and can then call the finite-difference level-set correction to make
the equality specification numerically precise.

The API deliberately does not hard-code ``PI``, ``CMem``, ``DD``, a network
matrix, or a particular resource cost.  For example, ``theta`` can be the full
entries of a VAR transition, while the objective may be Frobenius energy and a
soft penalty may enforce a stability margin using the existing
:func:`complextorch.linalg.spectral_radius` primitive.

This general Euclidean design routine is separate from SSDI optimization.
:func:`~complextorch.optimise_dynamical_dependence` solves a specialized
Grassmannian projection problem and retains its staged proxy--cluster--spectral
workflow.

Pareto selection
----------------

Functional degeneracy means that a prescribed capability vector need not select
one architecture.  The second design layer can therefore compare remaining
objectives over the feasible family:

.. math::

   \operatorname{Pareto}_{\theta\in\mathcal E_A(d^*)}
   [D_B(\theta),C(\theta),R(\theta),\ldots].

:func:`~complextorch.pareto_nondominated` returns the nondominated mask for a
set of candidate objective vectors and accepts a separate minimize/maximize
orientation for every objective.  It uses chunked comparisons to avoid
materializing the full ``N x N x M`` dominance tensor for large candidate sets.

The function filters a supplied set of designs; it does not claim to discover
or certify the global Pareto front.  Epsilon-constraint continuation,
preference-conditioned search, or other multiobjective optimizers can be built
by repeatedly using the same prescribed-capability and feasibility primitives.

Scientific scope and limitations
--------------------------------

The exact equations on this page assume stationary linear-Gaussian processes.
The modal representation additionally assumes simple, diagonalizable poles.
Differential rank, nullspaces, and level-set projection are local properties.
Optimization returns best-found solutions and does not establish global
optimality or global feasibility.  DD is defined relative to an explicit macro
projection ``L``; its scientific interpretation must always retain that
projection.

The design API is intentionally smaller than the research workflows that
motivated it.  Synthetic intervention families, particular four-node networks,
specific capability panels, and paper figures remain analysis artifacts rather
than package primitives.  This prevents the public API from encoding one
research narrative as a software contract.

References
----------

* Barnett, L. and Seth, A. K. (2015). Granger causality for state-space models.
  *Physical Review E*, 91, 040101.
* Barnett, L. and Seth, A. K. (2023). Dynamical independence: discovering
  emergent macroscopic processes in complex dynamical systems. *Physical
  Review E*, 108, 014304.
* Kailath, T. (1980). *Linear Systems*. Prentice-Hall.
* Trefethen, L. N. and Embree, M. (2005). *Spectra and Pseudospectra*.
  Princeton University Press.
