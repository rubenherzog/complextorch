Analytical measures
===================

ComplexTorch distinguishes model-derived analytical measures from observation
estimators. The model-derived layer consumes a supplied ``VARSystem``,
``StateSpaceModel``, or ``InnovationsStateSpace`` and does not silently refit
observations.

Gaussian entropy and mutual information
---------------------------------------

For :math:`X\sim\mathcal N(\mu,\Sigma)`, :math:`X\in\mathbb R^d`,

.. math::

   H(X)=\frac12\log_b\left[(2\pi e)^d\det\Sigma\right].

For jointly Gaussian :math:`X,Y`,

.. math::

   I(X;Y)
   =\frac12\log_b
   \frac{\det\Sigma_X\det\Sigma_Y}{\det\Sigma_{XY}}.

For jointly Gaussian :math:`X,Y,Z`,

.. math::

   I(X;Y\mid Z)
   =\frac12\log_b
   \frac{\det\Sigma_{XZ}\det\Sigma_{YZ}}
        {\det\Sigma_Z\det\Sigma_{XYZ}}.

Log determinants are evaluated through SPD-aware numerical routines.

Static high-order information
-----------------------------

For :math:`X=(X_1,\ldots,X_N)`, total correlation is

.. math::

   \mathrm{TC}(X)=\sum_{i=1}^NH(X_i)-H(X),

and dual total correlation is

.. math::

   \mathrm{DTC}(X)
   =\sum_{i=1}^NH(X_{-i})-(N-1)H(X),

where :math:`X_{-i}` denotes all variables except :math:`X_i`.

ComplexTorch defines O-information as

.. math::

   \Omega(X)=\mathrm{TC}(X)-\mathrm{DTC}(X),

and S-information as

.. math::

   S(X)=\mathrm{TC}(X)+\mathrm{DTC}(X).

Positive :math:`\Omega` indicates redundancy-dominated high-order dependence;
negative :math:`\Omega` indicates synergy-dominated dependence.

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

Multivariate Granger causality
------------------------------

Partition variables into target :math:`X`, source :math:`Y`, and conditioning
set :math:`Z`. Time-domain conditional MVGC is

.. math::

   F_{Y\to X\mid Z}
   =\log_b\frac{\det\Sigma^R_{XX}}{\det\Sigma_{XX}},

where :math:`\Sigma_{XX}` is the target innovation covariance in the full
process and :math:`\Sigma^R_{XX}` is the target innovation covariance after
removing source history. In the model-derived state-space path, the reduced
covariance is obtained by exact generalized-DARE marginalization.

The innovations transfer function is

.. math::

   H(z)=I+C(zI-A)^{-1}K.

ComplexTorch's state-space conditional spectral GC follows the Geweke/MVGC
construction and produces :math:`f_{Y\to X\mid Z}(\nu)`. On normalized
one-sided frequency :math:`\nu\in[0,1/2]`,

.. math::

   F_{Y\to X\mid Z}
   =2\int_0^{1/2}f_{Y\to X\mid Z}(\nu)\,d\nu.

This temporal/spectral identity is an important numerical validation check.
When ``conditional=None`` in the model-derived state-space API, ComplexTorch
uses the MVGC/ComplexBox convention of conditioning on all channels outside the
target and source groups. An explicitly empty conditioning set requests the
unconditioned calculation where supported.

Gaussian information rates
--------------------------

For stationary Gaussian blocks :math:`X,Y`, let :math:`V_X`, :math:`V_Y`, and
:math:`V_{XY}` be exact marginal/joint innovations covariances. Their mutual
information rate is

.. math::

   \dot I(X;Y)
   =\frac12\log_b\frac{|V_X||V_Y|}{|V_{XY}|}.

The corresponding spectral density is

.. math::

   i_{X;Y}(f)
   =\frac12\log_b
   \frac{|S_X(f)||S_Y(f)|}{|S_{XY}(f)|}.

For source :math:`S` and target :math:`T`, Gaussian transfer-entropy rate is

.. math::

   \dot T_{S\to T}
   =\frac12\log_b\frac{|V_T^R|}{|V_T|},

where :math:`V_T^R` is the exact target-only marginal innovation covariance and
:math:`V_T` is the target block of the joint source-target innovations
covariance. Thus, under the Gaussian convention,

.. math::

   \dot T_{S\to T}=\frac12F_{S\to T}.

For a joint innovations covariance

.. math::

   V=\begin{pmatrix}V_{XX}&V_{XY}\\V_{YX}&V_{YY}\end{pmatrix},

the instantaneous information-rate contribution is

.. math::

   \dot I_{X\circ Y}
   =\frac12\log_b\frac{|V_{XX}||V_{YY}|}{|V|}.

O-information rate
------------------

The O-information rate (OIR) extends static O-information from random variables
to stationary random processes. For :math:`N` process groups
:math:`X_1,\ldots,X_N`, ComplexTorch implements

.. math::

   \dot\Omega(X_1,\ldots,X_N)
   =(N-2)\dot H(X)
   +\sum_{i=1}^{N}\left[
      \dot H(X_i)-\dot H(X_{-i})
   \right].

For Gaussian innovations processes, entropy-rate constants cancel exactly. If
:math:`V`, :math:`V_i`, and :math:`V_{-i}` are the exact innovations
covariances of the full grouped process, group :math:`i`, and the leave-one-out
process, respectively, the implemented temporal identity is

.. math::

   \dot\Omega
   =\frac{1}{2\log b}\left[
      \sum_i\log|V_i|
      -\sum_i\log|V_{-i}|
      +(N-2)\log|V|
   \right].

Channels not listed in ``groups`` are marginalized out exactly through the
canonical innovations reduction. With two groups, OIR is identically zero.
Positive OIR is redundancy-dominated and negative OIR synergy-dominated.

The frequency-resolved OIR uses exact marginal spectral-density matrices:

.. math::

   \omega(f)
   =\frac{1}{2\log b}\left[
      \sum_i\log|S_i(f)|
      -\sum_i\log|S_{-i}(f)|
      +(N-2)\log|S(f)|
   \right].

Whole-band spectral integration recovers temporal OIR up to numerical
quadrature error.

O-information gradient / delta O-information rate
-------------------------------------------------

For selected group :math:`j`, ComplexTorch implements Faes et al.'s rate form

.. math::

   \Delta\dot\Omega_j
   =(2-N)\dot I(X_j;X_{-j})
   +\sum_{m\ne j}\dot I(X_j;X_{-\{j,m\}}).

This is exactly

.. math::

   \Delta\dot\Omega_j
   =\dot\Omega(X_1,\ldots,X_N)
   -\dot\Omega(X_1,\ldots,X_{j-1},X_{j+1},\ldots,X_N).

The temporal implementation is built from the independently validated Gaussian
MIR primitive rather than by subtracting two OIR evaluations. The spectral
version uses the spectral MIR primitive analogously, providing an independent
path for the defining identity.

Partial information rate decomposition (PIRD)
---------------------------------------------

ComplexTorch implements Gaussian partial information rate decomposition for
exactly two or three source groups and one disjoint target group, following the
Faes/HOP minimum-MIR convention.

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

The public spectral result exposes source-subset MIR spectra, redundancy
functions, all Möbius-inverted atoms, unique spectra per source, total redundant
and synergistic spectra, and

.. math::

   \Delta=\mathrm{redundant}-\mathrm{synergistic}.

For two sources, the coarse-grained atoms are unique source 1, unique source 2,
redundancy, and synergy. For three sources, the validated Faes/HOP coarse
graining combines the 18-node Williams--Beer lattice into three unique
components plus total redundant and synergistic components.

``partial_information_rate_decomposition`` integrates the spectral result. The
``half_open=True`` option follows the Faes/HOP half-open frequency-grid
convention and arithmetic-mean integration implemented by
``integrate_spectral_rate``; otherwise endpoint-inclusive trapezoidal
integration is used.

PIRD reuses the shared generalized-DARE reduction, spectral density, MIR,
integration, PID lattice, and Möbius inversion primitives rather than
implementing parallel numerical machinery.

PhiID
-----

For two variables and a positive lag :math:`\tau`, ComplexTorch constructs the
joint Gaussian covariance of

.. math::

   (X_t^1,X_t^2,X_{t+\tau}^1,X_{t+\tau}^2)

from model autocovariances. The Gaussian PhiID implementation uses the
minimum-mutual-information redundancy prescription on source and target
antichains drawn from

.. math::

   \{\mathrm{red},\mathrm{unq0},\mathrm{unq1},\mathrm{syn}\},

producing 16 source-to-target atoms. Möbius inversion yields the atoms and
satisfies

.. math::

   \sum_{\alpha,\beta}I_{\partial}^{\alpha\to\beta}
   =I\left((X_t^1,X_t^2);(X_{t+\tau}^1,X_{t+\tau}^2)\right).

Dynamical dependence and SSDI
-----------------------------

Let :math:`Y_t=LX_t`, with :math:`L\in\mathbb R^{m\times n}`. If
:math:`\Sigma` is the microscopic innovations covariance and :math:`\Sigma_R`
the exact innovations covariance of the projected process, ComplexTorch uses

.. math::

   F(X\to Y)
   =\log_b\frac{|\Sigma_R|}{|L\Sigma L^{\mathsf T}|}.

Under the Gaussian Shannon convention,

.. math::

   T(X\to Y)=\frac12F(X\to Y).

The DD function returns :math:`F`, not :math:`F/2`. The default log base is 2;
use natural logarithms for direct natural-log SSDI/ComplexBox comparison.

The unified optimizer is ``optimise_dynamical_dependence``. ``complexbox`` is
the default and recommended backend; ``riemannian_armijo`` is an opt-in
alternative. Optimization commonly uses row-orthonormal representatives

.. math::

   LL^{\mathsf T}=I_m,

but the scientific object is the projected subspace rather than a particular
basis.

Predictive emergence quantities
-------------------------------

For :math:`Y_t=MX_t`, ComplexTorch defines

.. math::

   \Psi=I(Y_t;X^-)-I(Y_t;Y^-),

.. math::

   \Delta=I(Y_t;Y^-)-\sum_jI(Y_t^j;Y_j^-),

.. math::

   \Gamma=I(Y_t;X^-)-\sum_jI(Y_t^j;X^-).

These quantities should be interpreted separately from SSDI dynamical
dependence even though both compare microscopic and macroscopic predictive
structure.

Criticality diagnostics
-----------------------

For transition spectral radius :math:`\rho`, ComplexTorch exposes

.. math::

   \text{stability margin}=1-\rho,

.. math::

   \tau=-\frac{\Delta t}{\log\rho},

and, for VAR systems,

.. math::

   A_{\mathrm{cov}}
   =\frac{\operatorname{tr}\Gamma_0}{\operatorname{tr}\Sigma}.

These are linear-system diagnostics and should not alone be interpreted as
evidence of a physical phase transition.

References
----------

- Cover, T. M. and Thomas, J. A. (2006). *Elements of Information Theory*.
- Geweke, J. (1982). Measurement of linear dependence and feedback between
  multiple time series. *JASA*, 77(378), 304--313.
- Barnett, L. and Seth, A. K. (2014). The MVGC toolbox. *Journal of
  Neuroscience Methods*, 223, 50--68.
- Barnett, L. and Seth, A. K. (2015). Granger causality for state-space models.
  *Physical Review E*, 91, 040101.
- Rosas, F. E. et al. (2019). Quantifying high-order interdependencies via the
  O-information. *Physical Review E*, 100, 032305.
- Williams, P. L. and Beer, R. D. (2010). Nonnegative decomposition of
  multivariate information. arXiv:1004.2515.
- Mediano, P. A. M. et al. (2021). Integrated information decomposition.
- Faes, L. et al. (2022). A new framework for the time- and frequency-domain
  assessment of high-order interactions in networks of random processes.
  *IEEE Transactions on Signal Processing*, 70, 5766--5777.
- Scagliarini, T. et al. (2023). Gradients of O-information: Low-order
  descriptors of high-order dependencies. *Physical Review Research*, 5,
  013025.
- Barnett, L. and Seth, A. K. (2023). Dynamical independence: Discovering
  emergent macroscopic processes in complex dynamical systems. *Physical
  Review E*, 108, 014304.

Repository references
---------------------

- ``src/complextorch/measures/gaussian.py``
- ``src/complextorch/measures/dynamics.py``
- ``src/complextorch/measures/mvgc.py``
- ``src/complextorch/measures/rates.py``
- ``src/complextorch/measures/oir.py``
- ``src/complextorch/measures/pird.py``
- ``src/complextorch/measures/_pid_lattice.py``
- ``src/complextorch/measures/phid.py``
- ``src/complextorch/measures/emergence.py``
- ``src/complextorch/measures/criticality.py``
- ``src/complextorch/control.py``
- ``src/complextorch/dd.py``
