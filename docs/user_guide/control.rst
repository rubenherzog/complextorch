Control, Riccati equations, and exact reductions
================================================

Steady-state filtering DARE
---------------------------

For

.. math::

   z_{t+1}=Az_t+w_t,
   \qquad
   y_t=Cz_t+v_t,

with :math:`\operatorname{Cov}(w_t)=Q` and
:math:`\operatorname{Cov}(v_t)=R`, ComplexTorch uses the filtering DARE

.. math::

   P=APA^{\mathsf T}+Q
   -APC^{\mathsf T}(CPC^{\mathsf T}+R)^{-1}CPA^{\mathsf T}.

:func:`~complextorch.solve_dare` exposes both the SciPy reference backend and
the Torch-native batched backend. ``backend="scipy"`` retains SciPy's
ordered-QZ solver; ``backend="torch"`` uses a device-native structured-doubling
algorithm.

In equivalent control-form notation, set

.. math::

   A_0=A^{\mathsf T},
   \qquad
   G_0=C^{\mathsf T}R^{-1}C,
   \qquad
   H_0=Q.

The structured-doubling iteration is

.. math::

   A_{k+1}=A_k(I+G_kH_k)^{-1}A_k,

.. math::

   G_{k+1}=G_k+A_kG_k(I+H_kG_k)^{-1}A_k^{\mathsf T},

.. math::

   H_{k+1}=H_k+A_k^{\mathsf T}H_k(I+G_kH_k)^{-1}A_k,

and :math:`H_k` converges to the stabilizing solution under the usual
stabilizability/detectability assumptions.

Generalized filtering DARE
--------------------------

If process and observation noise are correlated with

.. math::

   S=\operatorname{Cov}(w_t,v_t),

ComplexTorch uses

.. math::

   P=APA^{\mathsf T}+Q
   -(APC^{\mathsf T}+S)
   (CPC^{\mathsf T}+R)^{-1}
   (APC^{\mathsf T}+S)^{\mathsf T}.

:func:`~complextorch.solve_generalized_dare` implements this generalized
filtering equation. For positive-definite :math:`R`, its Torch backend performs
the exact noise-decorrelation transform

.. math::

   A_0=A-SR^{-1}C,

.. math::

   Q_0=Q-SR^{-1}S^{\mathsf T},

then reuses :func:`~complextorch.solve_dare` on :math:`(A_0,C,Q_0,R)`. This
avoids maintaining a second Riccati iteration.

Steady-state innovations form
-----------------------------

Given the standard DARE solution :math:`P`, the innovations covariance is

.. math::

   V=CPC^{\mathsf T}+R,

and the predictor-form gain is

.. math::

   K=APC^{\mathsf T}V^{-1}.

:func:`~complextorch.innovations_form` converts a
:class:`~complextorch.StateSpaceModel` to this steady-state representation,
while :func:`~complextorch.var_to_innovations_state_space` provides the exact
VAR bridge to :class:`~complextorch.InnovationsStateSpace`.

.. math::

   z_{t+1}=Az_t+K\varepsilon_t,
   \qquad
   y_t=Cz_t+\varepsilon_t,
   \quad
   \varepsilon_t\sim\mathcal N(0,V).

Exact projection of an innovations process
------------------------------------------

Consider the microscopic innovations model

.. math::

   z_{t+1}=Az_t+K\varepsilon_t,

.. math::

   X_t=Cz_t+\varepsilon_t,
   \qquad
   \operatorname{Cov}(\varepsilon_t)=\Sigma.

For a full-row-rank projection

.. math::

   Y_t=LX_t,
   \qquad L\in\mathbb R^{m\times n},

its observation matrix is

.. math::

   C_R=LC.

The state/process noise and projected observation noise are generated from the
same microscopic innovations. Therefore the exact reduced generalized
state-space covariances are

.. math::

   Q=K\Sigma K^{\mathsf T},

.. math::

   R=L\Sigma L^{\mathsf T},

.. math::

   S=K\Sigma L^{\mathsf T}.

The cross covariance :math:`S` is part of the exact projected process and must
not be discarded. :func:`~complextorch.reduce_innovations_state_space` performs
coordinate marginalization through this canonical generalized-DARE path.
:func:`~complextorch.project_state_space` applies an explicit linear observation
projection to a general state-space model, while
:func:`~complextorch.reduce_state_space` selects observation coordinates without
silently changing latent dynamics.

The generalized DARE produces the reduced prediction covariance :math:`P_R`,
after which

.. math::

   \Sigma_R=C_RP_RC_R^{\mathsf T}+R

is the exact innovations covariance of :math:`Y_t`. Coordinate marginalization
is the special case in which :math:`L` selects observation rows.

Innovations transfer function
-----------------------------

For an innovations model the transfer function from innovations to observations
is

.. math::

   H(z)=I+C(zI-A)^{-1}K,

with

.. math::

   z=e^{2\pi i f}

for normalized cycles per sample. :func:`~complextorch.innovations_transfer_function`
evaluates this response. :func:`~complextorch.innovations_spectral_density`
uses the same innovations model for spectral-density calculations, and
:func:`~complextorch.integrate_spectral_rate` provides the shared integration
path used by rate measures.

Why exact marginalization matters
---------------------------------

A marginal observation process of a finite-order VAR is generally not a VAR of
the same finite order. ComplexTorch therefore uses
:class:`~complextorch.InnovationsStateSpace` reduction plus
:func:`~complextorch.solve_generalized_dare` for model-derived marginal
quantities instead of silently truncating a marginal VAR. This convention is
central to state-space Granger causality, information-rate decompositions, and
:func:`~complextorch.dynamical_dependence`.

References
----------

- Anderson, B. D. O. and Moore, J. B. (1979). *Optimal Filtering*.
- Laub, A. J. (1979). A Schur method for solving algebraic Riccati equations.
- van Dooren, P. (1981). A generalized eigenvalue approach for solving Riccati
  equations.
- Barnett, L. and Seth, A. K. (2015). Granger causality for state-space models.
  *Physical Review E*, 91, 040101.

Repository references
---------------------

- ``src/complextorch/control.py``
- ``src/complextorch/linalg.py``
- ``src/complextorch/representations.py``
