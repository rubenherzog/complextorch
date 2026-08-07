Vector autoregressive models
============================

Model definition
----------------

ComplexTorch uses the Gaussian VAR(:math:`p`) model

.. math::

   x_t = c + \sum_{k=1}^{p} A_k x_{t-k} + \varepsilon_t,
   \qquad
   \varepsilon_t \sim \mathcal N(0,\Sigma),

where :math:`x_t\in\mathbb R^n`, :math:`c\in\mathbb R^n`,
:math:`A_k\in\mathbb R^{n\times n}`, and :math:`\Sigma` is the innovations
covariance. Coefficients use the ``(lag, target, source)`` convention for a
single system and ``(batch, lag, target, source)`` when batched.

Least-squares estimation
------------------------

Given a lagged design matrix :math:`X` and targets :math:`Y`, ordinary least
squares solves

.. math::

   \widehat B = \arg\min_B \|Y-XB\|_F^2.

With ridge strength :math:`\alpha\ge0`, the penalized problem is

.. math::

   \widehat B
   = \arg\min_B\left(\|Y-XB\|_F^2+\alpha\|B\|_F^2\right),

with the fitted intercept excluded from the ridge penalty. ComplexTorch exposes
``lstsq``, Cholesky, pseudoinverse, and LWR/Morf-style solvers. ``solver="auto"``
uses least squares for zero ridge and Cholesky for ridge regression.

Innovation covariance and likelihood
------------------------------------

For residuals :math:`e_t=x_t-\widehat x_t`, the MLE covariance is

.. math::

   \widehat\Sigma_{\mathrm{MLE}}
   = \frac1N\sum_{t=1}^{N} e_t e_t^{\mathsf T}.

The alternative ``unbiased`` convention uses an adjusted denominator based on
effective observations and fitted predictors. The covariance convention should
be reported in reproducible analyses and parity studies.

The Gaussian negative log likelihood for one residual is

.. math::

   \operatorname{NLL}_t
   = \frac12\left[
       e_t^{\mathsf T}\Sigma^{-1}e_t
       + \log\det\Sigma
       + n\log(2\pi)
     \right].

The implementation evaluates the quadratic term by Cholesky solve rather than
forming :math:`\Sigma^{-1}` explicitly.

Companion representation
------------------------

Define the companion state

.. math::

   z_t =
   \begin{bmatrix}
   x_t\\x_{t-1}\\\vdots\\x_{t-p+1}
   \end{bmatrix}
   \in\mathbb R^{np}.

Then

.. math::

   z_t = A_c z_{t-1} + \eta_t,

with

.. math::

   A_c =
   \begin{pmatrix}
   A_1&A_2&\cdots&A_p\\
   I&0&\cdots&0\\
   0&I&\cdots&0\\
   \vdots&&\ddots&\vdots
   \end{pmatrix}.

The companion process-noise covariance has the innovations covariance in its
leading block,

.. math::

   Q_c =
   \begin{pmatrix}
   \Sigma&0\\0&0
   \end{pmatrix}.

For a stable VAR, the stationary companion covariance :math:`P` solves

.. math::

   P=A_cPA_c^{\mathsf T}+Q_c.

Let :math:`C_p=[I,0,\ldots,0]` select the present observation block. Then

.. math::

   \Gamma_0=C_pPC_p^{\mathsf T},

and

.. math::

   \Gamma_k=C_pA_c^kPC_p^{\mathsf T}.

The companion representation is the canonical bridge between fitted VAR
parameters and the analytical covariance, spectral, information-theoretic, and
state-space layers.

Forecasting
-----------

Recursive forecasts use

.. math::

   \widehat x_{t+h\mid t}
   =c+\sum_{k=1}^{p}A_k\widehat x_{t+h-k\mid t},

where observed values initialize the recursion and previously predicted values
are used once the requested horizon extends beyond the available history.

Stability and diagnostics
-------------------------

Stationarity requires

.. math::

   \rho(A_c)<1.

ComplexTorch exposes the fitted spectral radius and stability flag and also
provides residual consistency and whiteness diagnostics. These diagnostics
should be examined before interpreting model-derived causal or information
measures.

References
----------

- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*.
- Morf, M., Vieira, A., Lee, D. T. L., and Kailath, T. (1978). Recursive
  multichannel maximum-entropy spectral estimation.
- Barnett, L. and Seth, A. K. (2014). The MVGC multivariate Granger causality
  toolbox. *Journal of Neuroscience Methods*, 223, 50--68.

Repository references
---------------------

- ``src/complextorch/var.py``
- ``src/complextorch/representations.py``
- ``src/complextorch/measures/dynamics.py``
- ``src/complextorch/measures/secondary.py``
