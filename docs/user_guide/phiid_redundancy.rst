PhiID redundancy prescriptions
==============================

Model-first contract
--------------------

ComplexTorch evaluates bivariate Gaussian PhiID from a supplied generative
model. For observed variables :math:`X_1,X_2` and lag :math:`\tau`, the model
autocovariance backbone constructs

.. math::

   \Sigma_{PF}=\operatorname{Cov}
   \left(X_{1,t},X_{2,t},X_{1,t+\tau},X_{2,t+\tau}\right).

``phiid_redundancy_from_model`` passes this exact model-derived covariance to
``gaussian_phiid_atoms``. No observations are refitted and no empirical
covariance is estimated inside the PhiID calculation.

The default remains ``redundancy="mmi"`` for backward compatibility. Available
continuous-Gaussian backends are ``mmi``, ``ccs``, ``idep_a``, and ``idep_b``.

The 16 atoms
------------

For source modes and target modes in

.. math::

   \{\mathrm{red},\mathrm{unq0},\mathrm{unq1},\mathrm{syn}\},

PhiID contains 16 source-to-target atoms. ComplexTorch follows the bivariate
``15-for-free`` construction used by the Imperial MIND-lab reference code. The
known cumulative quantities are

.. math::

   rtr,
   R_{xy\to a},R_{xy\to b},R_{xy\to ab},
   R_{ab\to x},R_{ab\to y},R_{ab\to xy},

followed by the nine ordinary mutual informations

.. math::

   I(x;a), I(x;b), I(y;a), I(y;b),
   I(xy;a), I(xy;b), I(x;ab), I(y;ab), I(xy;ab).

These 16 cumulative terms are mapped to the 16 atoms by the fixed linear system
published in the reference implementation. The reconstruction identity is

.. math::

   \sum_{\alpha,\beta} I_{\partial}^{\alpha\to\beta}
   = I((x,y);(a,b)).

MMI
---

For two predictors :math:`A,B` and target :math:`T`, minimum mutual
information redundancy is

.. math::

   I_{\mathrm{MMI}}(A,B;T)
   = \min\{I(A;T),I(B;T)\}.

The MMI double-redundancy is

.. math::

   rtr_{\mathrm{MMI}}
   =\min\{I(x;a),I(x;b),I(y;a),I(y;b)\}.

All Gaussian mutual informations are evaluated analytically from covariance
log-determinants. The new implementation is regression-tested against the
previous ComplexTorch MMI product-lattice calculation.

CCS
---

Common change in surprisal (CCS) is defined pointwise. Let

.. math::

   i_x=i(A;T),\qquad
   i_y=i(B;T),\qquad
   i_{xy}=i((A,B);T),

and

.. math::

   c=i_x+i_y-i_{xy}.

The local CCS contribution is :math:`c` only when :math:`i_x`, :math:`i_y`,
:math:`i_{xy}`, and :math:`c` have the same sign; otherwise it is zero.

For PhiID double redundancy, ComplexTorch follows exactly the pinned Imperial
MIND-lab construction. Define the local quantity

.. math::

   c_2 =
   -i_{x;a}-i_{x;b}-i_{y;a}-i_{y;b}
   +i_{x;ab}+i_{y;ab}+i_{xy;a}+i_{xy;b}-i_{xy;ab}

.. math::

   \qquad
   +R_{xy\to a}+R_{xy\to b}-R_{xy\to ab}
   +R_{ab\to x}+R_{ab\to y}-R_{ab\to xy}.

A local sample contributes :math:`c_2` only when the signs of
:math:`i_{x;a}`, :math:`i_{x;b}`, :math:`i_{y;a}`, :math:`i_{y;b}`, and
:math:`c_2` agree.

CCS has no closed-form Gaussian expectation in the cited implementation.
ComplexTorch therefore performs deterministic Sobol quasi-Monte-Carlo
integration under :math:`\mathcal N(0,\Sigma_{PF})`. The default is 4096 nodes.
These nodes are numerical quadrature points generated from the model parameters;
they are not observations and are never used to refit model parameters.

Gaussian dependency-constraint PID
----------------------------------

For predictors :math:`X_0,X_1` and target :math:`Y`, block-whiten the covariance
into

.. math::

   \widetilde\Sigma=
   \begin{pmatrix}
      I & P & Q\\
      P^{\mathsf T} & I & R\\
      Q^{\mathsf T} & R^{\mathsf T} & I
   \end{pmatrix}.

ComplexTorch obtains :math:`P,Q,R` by Cholesky triangular solves; it does not
form explicit covariance inverses. Following Kay and Ince's Gaussian
implementation, the candidate unique-information edges for :math:`X_0` are

.. math::

   b=I(X_0;Y),

.. math::

   i=\frac12\log_b|I-RQ^{\mathsf T}QR^{\mathsf T}|
     -\frac12\log_b|I-Q^{\mathsf T}Q|
     -\frac12\log_b|I-R^{\mathsf T}R|-I(X_1;Y),

and

.. math::

   k=\frac12\log_b|I-P^{\mathsf T}P|
     -\frac12\log_b|\widetilde\Sigma|-I(X_1;Y).

Then

.. math::

   \mathrm{Unq}_0=\min\{b,i,k\},\qquad
   \mathrm{Red}=I(X_0;Y)-\mathrm{Unq}_0.

This single-target Gaussian I_dep calculation is pinned to
``robince/partial-info-decomp`` commit
``32207164741b9e3ba86cec225c09b4b617681e93`` and protected by an independent
numerical fixture.

``idep_a`` and ``idep_b``
-------------------------

The archived ComplexBox ELPH port supplied for this work and the pinned Imperial
MIND-lab implementation expose MMI and CCS, but do not expose methods named
``Idep_a`` or ``Idep_b``. Consequently ComplexTorch does not claim external
naming parity for these two labels.

The single-target I_dep prescription fixes 15 PhiID cumulative terms but leaves
the bottom double-redundancy to be closed. ComplexTorch exposes both directional
closures explicitly:

``idep_a``
   The bottom double-redundancy is the minimum of the two forward single-target
   I_dep redundancies, one for each future target.

``idep_b``
   The bottom double-redundancy is the minimum of the two time-reversed
   single-target I_dep redundancies, one for each past target.

They are tested as time-reversal duals. If an authoritative ELPH source is later
provided that assigns these names differently, that source should replace this
naming convention and parity should be revalidated.

Varley's shared-exclusion redundancy
------------------------------------

Varley (2023) uses :math:`I_{\tau sx}` as a PhiID redundancy based on shared
probability-mass exclusions. The paper explicitly states that the construction
is currently well-defined only for discrete random variables and that a
continuous generalization remains open.

ComplexTorch therefore does not silently discretize a Gaussian model and does
not invent a continuous :math:`I_{\tau sx}`. Passing ``redundancy="varley"``
or an :math:`I_{\tau sx}` alias to the Gaussian backend raises
``NotImplementedError``. A future implementation requires a primary discrete
generative-distribution contract that supplies the probability mass function
from model parameters.

Numerics and validation
-----------------------

The implementation preserves leading batch dimensions, dtype, and device.
Cholesky factorization, triangular solves, and ``torch.linalg.solve`` are used
instead of explicit inverses. DEBUG logging records the selected backend,
block size, logarithm base, dtype/device, CCS quadrature size when applicable,
and the maximum 16-atom reconstruction residual.

Regression tests cover MMI backward compatibility, the pinned Imperial CCS
double-redundancy equations, the pinned Kay--Ince Gaussian I_dep fixture,
batch-vs-loop equivalence for every backend, float32/float64 agreement for
analytic backends, model-first equivalence, input validation, time reversal,
and CUDA parity when CUDA is available.

References
----------

- Mediano, P. A. M. et al. (2021). *Towards an extended taxonomy of
  information dynamics via Integrated Information Decomposition*.
- Ince, R. A. A. (2017). *Entropy* 19, 318.
- Kay, J. W. and Ince, R. A. A. (2018). Exact partial information
  decompositions for Gaussian systems based on dependency constraints.
- James, R. G., Emenheiser, J., and Crutchfield, J. P. (2019). Unique
  information via dependency constraints. *Journal of Physics A* 52, 014002.
- Varley, T. F. (2023). Decomposing past and future: Integrated information
  decomposition based on shared exclusions. *PLOS ONE* 18, e0282950.

Pinned software references
--------------------------

- ``Imperial-MIND-lab/integrated-info-decomp`` commit
  ``6c5f2e9d33c985efbdf875d45cb5a2a6a5cdbf44``.
- ``robince/partial-info-decomp`` commit
  ``32207164741b9e3ba86cec225c09b4b617681e93``.
- Uploaded ComplexBox archive commit
  ``87b5e2cd9bba22ddd978bade6f614da7d6190db2``.
