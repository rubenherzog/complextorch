Dynamics scaling
================

:func:`~complextorch.scale_dynamics` constructs a one-parameter family around a
fitted linear Gaussian dynamical system without refitting observations.  The
operation is defined on a common innovations representation, so the same
transformation applies to :class:`~complextorch.VARSystem`,
:class:`~complextorch.StateSpaceModel`, and
:class:`~complextorch.InnovationsStateSpace` inputs.

Canonical innovations representation
------------------------------------

Use :func:`~complextorch.as_innovations_state_space` when an explicit common
representation is useful.  It maps supported systems to

.. math::

   x_{t+1}=Ax_t+K\varepsilon_t,
   \qquad
   y_t=Cx_t+\varepsilon_t,
   \qquad
   \operatorname{cov}(\varepsilon_t)=V.

For an innovations model, the equivalent correlated-noise generative
quantities are

.. math::

   Q=KVK^{\mathsf T},
   \qquad
   R=V,
   \qquad
   S=KV.

Here :math:`S=\operatorname{cov}(w_t,v_t)` is the process--observation cross
covariance.  These quantities are retained by the scaling transformation.

Scaling parameter
-----------------

The scaled family is defined by

.. math::

   A_\lambda=\lambda A,
   \qquad \lambda\geq0,

while :math:`C,Q,R,S` are held fixed.  The generalized steady-state DARE is
then solved to obtain the exact innovations gain :math:`K_\lambda` and
innovation covariance :math:`V_\lambda` of the deformed process.  The returned
object is therefore again an :class:`~complextorch.InnovationsStateSpace`.

The control parameter has three useful reference points:

.. math::

   \lambda=0
   \quad\Longrightarrow\quad
   A_\lambda=0,

which removes latent temporal propagation;

.. math::

   \lambda=1,

which recovers the empirical model up to numerical Riccati tolerance; and, for
an empirical system with spectral radius :math:`\rho_0>0`,

.. math::

   \lambda_c=\frac{1}{\rho_0},

which is the stability boundary.  Since

.. math::

   \rho(\lambda)=\lambda\rho_0,

one can describe the same family using the relative scale :math:`\lambda`, the
absolute spectral radius :math:`\rho`, or the distance to instability

.. math::

   \epsilon=1-\rho.

The exact boundary :math:`\lambda_c` is excluded because the stationary
covariance-based measures in ComplexTorch require strict stability.

Why scale in innovations form?
------------------------------

The transformation is deliberately defined after canonical conversion instead
of introducing separate VAR and state-space scaling rules.  Multiplying the
transition matrix by a scalar preserves its eigenvectors and moves every pole
radially by the same factor.  It is also invariant under a change of latent
coordinates: if :math:`A'=TAT^{-1}`, then

.. math::

   \lambda A'=T(\lambda A)T^{-1}.

Thus the deformation does not depend on the particular state-space basis.
Keeping :math:`C,Q,R,S` fixed while recomputing the innovations quantities
preserves the stochastic and observational architecture of the model rather
than treating the original :math:`K` and :math:`V` as fixed after the dynamics
have changed.

Basic usage
-----------

A single scaled model can be generated directly from any supported canonical
system:

.. code-block:: python

   from complextorch import scale_dynamics

   scaled = scale_dynamics(system, 0.8)

Explicit conversion is optional:

.. code-block:: python

   from complextorch import as_innovations_state_space, scale_dynamics

   innovations = as_innovations_state_space(system)
   scaled = scale_dynamics(innovations, 0.8)

A one-dimensional tensor is interpreted as a common grid and is applied to
every input batch element:

.. code-block:: python

   import torch
   from complextorch import scale_dynamics

   lambdas = torch.tensor([0.0, 0.5, 1.0, 1.1], dtype=torch.float64)
   family = scale_dynamics(system, lambdas)

For a batch of ``B`` systems and ``L`` lambda values, the returned innovations
model contains ``B * L`` systems in system-major order.  This layout is intended
for direct downstream use with batched model-based measures.

Choosing a stable grid
----------------------

For a stable empirical model with spectral radius :math:`\rho_0`, values must
satisfy

.. math::

   \lambda\rho_0<1.

For example, a near-critical point with target spectral radius
:math:`\rho_*=0.99` is obtained from

.. math::

   \lambda_*=\frac{0.99}{\rho_0}.

The empirical point remains exactly :math:`\lambda=1`, while the same curve can
be compared across systems using the derived absolute spectral radius or
:math:`\epsilon=1-\rho`.

Notes
-----

``scale_dynamics`` is an analytical transformation of an already supplied
model.  It does not perform model selection, refit the original observations,
or change the criterion used to choose the empirical model.
