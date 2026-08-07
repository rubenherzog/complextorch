Mathematical and tensor conventions
===================================

Notation
--------

For VAR models, observations are written :math:`x_t\in\mathbb R^n`. For
state-space models, observations and latent states are written
:math:`y_t\in\mathbb R^n` and :math:`z_t\in\mathbb R^r`. Here ``n`` is the
observation dimension, ``r`` the latent-state dimension, ``p`` the VAR lag
order, and ``t`` the discrete-time index. VAR lag order and latent state
dimension are distinct quantities and are selected by different procedures.

For a zero-mean stationary process, the lag-:math:`k` autocovariance is

.. math::

   \Gamma_k = \operatorname{Cov}(x_t,x_{t-k})
             = \mathbb E[x_t x_{t-k}^{\mathsf T}].

For a block Gaussian covariance

.. math::

   \Sigma =
   \begin{pmatrix}
   \Sigma_{XX} & \Sigma_{XY}\\
   \Sigma_{YX} & \Sigma_{YY}
   \end{pmatrix},

ComplexTorch uses the Schur-complement conditional covariance

.. math::

   \Sigma_{X\mid Y}
   = \Sigma_{XX}
   - \Sigma_{XY}\Sigma_{YY}^{-1}\Sigma_{YX},

but evaluates the corresponding linear solve rather than explicitly forming the
inverse.

Stability
---------

For a first-order transition matrix :math:`A`, or a VAR companion matrix
:math:`A_c`, stability requires

.. math::

   \rho(A) = \max_i |\lambda_i(A)| < 1.

For VAR models the companion construction is exposed by
:func:`~complextorch.companion_matrix`, while
:class:`~complextorch.VARSystem` stores the corresponding stability information.

For a stable dominant mode, the discrete decay timescale used by ComplexTorch is

.. math::

   \tau = -\frac{\Delta t}{\log\rho},

with :math:`\Delta t=1` unless another sampling interval is supplied.

Input shapes and trajectory semantics
-------------------------------------

Time-series estimators such as :class:`~complextorch.VAR`,
:class:`~complextorch.N4SID`, and :class:`~complextorch.LarimoreStateSpace`
accept either ``(time, variables)`` or ``(batch, time, variables)`` observations.
Thus a batched data tensor has shape

.. math::

   (B,T,n),

where ``B`` counts trajectories, ``T`` is the number of samples per trajectory,
and ``n`` is the number of observed variables.

``mode="independent"`` estimates one model per trajectory,

.. math::

   X^{(b)} \longrightarrow \theta^{(b)},\qquad b=1,\ldots,B.

``mode="pooled"`` assumes independent realizations of one common model,

.. math::

   \theta^{(1)}=\cdots=\theta^{(B)}=\theta,

but only pools valid within-trajectory sufficient statistics. ComplexTorch must
never create a lag, state transition, Hankel block, residual pair, or validation
relationship across a trajectory boundary.

For example, a VAR lag design is first formed separately for every trajectory,
and only then may valid regression rows be concatenated. The same rule governs
:class:`~complextorch.LarimoreStateSpace`/:class:`~complextorch.N4SID`
block-Hankel matrices, :class:`~complextorch.LinearGaussianEM` transition
statistics, and temporal cross-validation through
:class:`~complextorch.EpochTimeSeriesSplit`.

Coefficient orientation
-----------------------

:class:`~complextorch.VAR` coefficients are stored as
``(batch, lag, target, source)``. For a single system the natural shape is
``(p, n, n)``, and :math:`A_k[i,j]` represents the lag-:math:`k` contribution
of source ``j`` to target ``i``.

Canonical representations
-------------------------

ComplexTorch intentionally distinguishes three representations:

:class:`~complextorch.VARSystem`
   A stationary Gaussian VAR together with its companion transition, companion
   noise covariance, stationary state covariance, observation projection, and
   stability information. :func:`~complextorch.build_var_system` constructs this
   canonical representation from coefficients and innovations covariance.

:class:`~complextorch.StateSpaceModel`
   A general linear Gaussian state-space model parameterized by
   :math:`(A,C,Q,R)`.

:class:`~complextorch.InnovationsStateSpace`
   An innovations-form model parameterized by :math:`(A,C,K,V)`. General models
   can be converted through :func:`~complextorch.innovations_form`, and VAR
   systems through :func:`~complextorch.var_to_innovations_state_space`.

A general state-space model and an innovations model are not interchangeable:

.. math::

   (A,C,Q,R) \neq (A,C,K,V)

in general.

References
----------

- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*.
- Anderson, B. D. O. and Moore, J. B. (1979). *Optimal Filtering*.
- Cover, T. M. and Thomas, J. A. (2006). *Elements of Information Theory*.

Repository references
---------------------

- ``src/complextorch/representations.py``
- ``src/complextorch/var.py``
- ``src/complextorch/state_space.py``
- ``src/complextorch/_subspace.py``
- ``src/complextorch/selection/_temporal.py``
