State-space models and estimation
=================================

General linear Gaussian state-space model
-----------------------------------------

:class:`~complextorch.StateSpaceModel` represents

.. math::

   z_{t+1}=Az_t+w_t,
   \qquad w_t\sim\mathcal N(0,Q),

.. math::

   y_t=Cz_t+v_t,
   \qquad v_t\sim\mathcal N(0,R),

where :math:`z_t\in\mathbb R^r` is the latent state and
:math:`y_t\in\mathbb R^n` the observation. The model parameters are
:math:`A\in\mathbb R^{r\times r}`, :math:`C\in\mathbb R^{n\times r}`,
:math:`Q\in\mathbb R^{r\times r}`, and :math:`R\in\mathbb R^{n\times n}`.
When available, the stationary latent covariance :math:`P_z` satisfies

.. math::

   P_z=AP_zA^{\mathsf T}+Q.

Innovations representation
--------------------------

:class:`~complextorch.InnovationsStateSpace` represents the
predictor/innovations form

.. math::

   z_{t+1}=Az_t+K\varepsilon_t,

.. math::

   y_t=Cz_t+\varepsilon_t,
   \qquad \varepsilon_t\sim\mathcal N(0,V).

Here :math:`K` is the innovations gain and :math:`V` the innovations covariance.
A general model is converted with :func:`~complextorch.innovations_form`; a
:class:`~complextorch.VARSystem` is converted with
:func:`~complextorch.var_to_innovations_state_space`. Exact marginalization uses
:func:`~complextorch.reduce_innovations_state_space`.

Kalman filtering
----------------

:func:`~complextorch.kalman_filter` implements the linear-Gaussian filter. Let
:math:`m_t^-` and :math:`P_t^-` denote the one-step predicted state moments. The
prediction equations are

.. math::

   m_t^-=Am_{t-1},

.. math::

   P_t^-=AP_{t-1}A^{\mathsf T}+Q.

The observation innovation and covariance are

.. math::

   e_t=y_t-Cm_t^-,

.. math::

   S_t=CP_t^-C^{\mathsf T}+R.

The Kalman gain is

.. math::

   K_t=P_t^-C^{\mathsf T}S_t^{-1},

and the posterior mean is

.. math::

   m_t=m_t^-+K_te_t.

ComplexTorch uses the Joseph covariance update

.. math::

   P_t=(I-K_tC)P_t^-(I-K_tC)^{\mathsf T}+K_tRK_t^{\mathsf T},

which is algebraically equivalent to the usual covariance update but better
preserves symmetry and positive semidefiniteness in finite precision.

The exact Gaussian log likelihood accumulated by the filter is

.. math::

   \log p(y_{1:T})
   =-\frac12\sum_t\left[
      n\log(2\pi)+\log\det S_t+e_t^{\mathsf T}S_t^{-1}e_t
   \right].

Rauch--Tung--Striebel smoothing
-------------------------------

:func:`~complextorch.kalman_smoother` applies the RTS backward pass. The RTS
backward gain is

.. math::

   J_t=P_{t\mid t}A^{\mathsf T}(P_{t+1\mid t})^{-1}.

The smoothed mean is

.. math::

   m_{t\mid T}=m_{t\mid t}
   +J_t\left(m_{t+1\mid T}-m_{t+1\mid t}\right),

and the smoothed covariance is

.. math::

   P_{t\mid T}=P_{t\mid t}
   +J_t\left(P_{t+1\mid T}-P_{t+1\mid t}\right)J_t^{\mathsf T}.

Block-Hankel convention
-----------------------

For subspace identification, :class:`~complextorch.N4SID`,
:class:`~complextorch.LarimoreStateSpace`, and the state-space selectors build
past and future block-Hankel matrices inside each trajectory. For a time origin
:math:`t`, the past block is ordered recent-to-distant,

.. math::

   P_t=\begin{bmatrix}
   y_{t-1}\\y_{t-2}\\\vdots\\y_{t-h_p}
   \end{bmatrix},

while the future block begins at the current sample,

.. math::

   F_t=\begin{bmatrix}
   y_t\\y_{t+1}\\\vdots\\y_{t+h_f-1}
   \end{bmatrix}.

No column may span a trajectory boundary.

N4SID
-----

The compact :class:`~complextorch.N4SID` implementation forms a regularized
future-on-past projection

.. math::

   \Pi_{F\mid P}
   =FP^{\mathsf T}
   (PP^{\mathsf T}+\lambda I)^{\dagger}P.

With singular-value decomposition

.. math::

   \Pi_{F\mid P}=USV^{\mathsf T},

and requested state dimension :math:`r`, the retained observability basis is

.. math::

   \mathcal O_r=U_rS_r^{1/2}.

State columns are then obtained by solving

.. math::

   \mathcal O_r X\approx F.

Given estimated states, ComplexTorch fits

.. math::

   z_{t+1}\approx Az_t,
   \qquad
   y_t\approx Cz_t,

by least squares and estimates :math:`Q` and :math:`R` from the corresponding
residual covariances. In pooled mode, transition pairs are pooled only after
within-trajectory states have been constructed.

Larimore canonical variate analysis
-----------------------------------

:class:`~complextorch.LarimoreStateSpace` estimates the innovations model

.. math::

   z_{t+1}=Az_t+K\varepsilon_t,
   \qquad
   y_t=Cz_t+\varepsilon_t,
   \quad
   \varepsilon_t\sim\mathcal N(0,V).

Let :math:`P` and :math:`F` denote past and future Hankel matrices with
:math:`N` valid columns. ComplexTorch forms

.. math::

   \Sigma_{PP}=PP^{\mathsf T}/N,
   \qquad
   \Sigma_{FF}=FF^{\mathsf T}/N,
   \qquad
   \Sigma_{FP}=FP^{\mathsf T}/N.

Whitening uses

.. math::

   L_PL_P^{\mathsf T}=\Sigma_{PP}+\lambda I,
   \qquad
   L_FL_F^{\mathsf T}=\Sigma_{FF}+\lambda I,

followed by

.. math::

   M=L_F^{-1}\Sigma_{FP}L_P^{-\mathsf T}.

If

.. math::

   M=U\operatorname{diag}(\rho_1,\ldots,\rho_q)V^{\mathsf T},

then :math:`\rho_i` are the Larimore canonical correlations. For a selected
state dimension :math:`r`, the CVA state sequence is constructed from the
whitened past canonical variates as

.. math::

   Z = \operatorname{diag}(\rho_1,\ldots,\rho_r)
       V_r^{\mathsf T} L_P^{-1} P.

This normalization gives the whitened past :math:`L_P^{-1}P`, whose covariance
is the identity (up to the optional ridge regularization). The fixed-order
estimator then fits :math:`C` from every valid state/observation pair, :math:`A`
from only within-trajectory state transitions, and :math:`K` from

.. math::

   z_{t+1}-Az_t\approx K\varepsilon_t.

The innovations covariance is the covariance of

.. math::

   \varepsilon_t=y_t-Cz_t.

Use :class:`~complextorch.StateSpaceOrderSelection` for Bauer/Larimore latent
dimension selection, or :class:`~complextorch.StateSpaceOrderSearchCV` for
temporal cross-validation of state dimension. Selection remains separate from
the fixed-order :class:`~complextorch.LarimoreStateSpace` estimator.

EM refinement
-------------

:class:`~complextorch.LinearGaussianEM` performs expectation--maximization for a
general :math:`(A,C,Q,R)` model. Its E-step calls
:func:`~complextorch.kalman_smoother` to obtain

.. math::

   \mathbb E[z_t\mid y_{1:T}],\quad
   \mathbb E[z_tz_t^{\mathsf T}\mid y_{1:T}],\quad
   \mathbb E[z_tz_{t-1}^{\mathsf T}\mid y_{1:T}].

Define

.. math::

   S_{00}=\sum_t\mathbb E[z_tz_t^{\mathsf T}],\qquad
   S_{11}=\sum_t\mathbb E[z_{t+1}z_{t+1}^{\mathsf T}],

.. math::

   S_{10}=\sum_t\mathbb E[z_{t+1}z_t^{\mathsf T}].

The transition update is

.. math::

   A_{\mathrm{new}}=S_{10}S_{00}^{\dagger},

and the process covariance update is

.. math::

   Q_{\mathrm{new}}
   =\frac1N\left(
   S_{11}-AS_{10}^{\mathsf T}-S_{10}A^{\mathsf T}+AS_{00}A^{\mathsf T}
   \right).

The observation update is

.. math::

   C_{\mathrm{new}}
   =\left(\sum_t y_t\mathbb E[z_t]^{\mathsf T}\right)
    \left(\sum_t\mathbb E[z_tz_t^{\mathsf T}]\right)^{\dagger}.

Covariance updates are symmetrized and floored by the configured numerical
minimum. In pooled mode, sufficient statistics sum over trajectory and time but
never include cross-trajectory transitions.

References
----------

- Kalman, R. E. (1960). A new approach to linear filtering and prediction.
- Rauch, H. E., Tung, F., and Striebel, C. T. (1965). Maximum-likelihood
  estimates of linear dynamic systems.
- Van Overschee, P. and De Moor, B. (1994). N4SID. *Automatica*, 30(1), 75--93.
- Larimore, W. E. (1990, 1996). Canonical variate analysis system
  identification.
- Shumway, R. H. and Stoffer, D. S. (1982). EM for time-series smoothing and
  state-space estimation.

Repository references
---------------------

- ``src/complextorch/state_space.py``
- ``src/complextorch/_subspace.py``
- ``src/complextorch/representations.py``
