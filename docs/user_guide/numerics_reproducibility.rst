Numerical conventions and reproducibility
==========================================

Torch-first numerical contract
------------------------------

ComplexTorch treats numerical behavior as part of the scientific API. Where
practical, calculations remain in Torch and preserve dtype, device, and leading
batch dimensions. NumPy/SciPy conversion is reserved for explicitly documented
reference backends such as ``solve_dare(..., backend="scipy")``.

Linear solves instead of explicit inverses
------------------------------------------

Expressions written mathematically as

.. math::

   A^{-1}B

are normally evaluated by solving

.. math::

   AX=B.

Preferred primitives include ``torch.linalg.solve``,
``torch.linalg.solve_triangular``, ``torch.linalg.lstsq``, Cholesky solves, QR,
and SVD where required by the mathematics.

Symmetric positive-definite matrices
------------------------------------

For :math:`\Sigma=LL^{\mathsf T}` with Cholesky factor :math:`L`,

.. math::

   \log\det\Sigma=2\sum_i\log L_{ii}.

Covariance-like outputs are explicitly symmetrized where appropriate,

.. math::

   \operatorname{sym}(M)=\frac12(M+M^{\mathsf T}).

Small covariance floors or adaptive jitter are numerical stabilization devices,
not hidden statistical model assumptions. Their use should be documented when
scientifically material.

Working precision
-----------------

Public dtype is normally preserved. A documented exception is the Torch DARE
backend: float32 inputs use float64 working precision on the same device and are
cast back to float32 after convergence. The generalized Torch DARE inherits the
same policy because it decorrelates the noises and reuses the standard solver.

Complex frequency-domain quantities
-----------------------------------

Float64 real dynamics naturally produce complex128 spectral quantities; float32
dynamics produce complex64. Spectral-density matrices should satisfy

.. math::

   S(f)=S(f)^*.

Finite-precision code may explicitly project onto the Hermitian part

.. math::

   \operatorname{Herm}(S)=\frac12(S+S^*).

Frequency conventions
---------------------

Frequency arrays are interpreted in cycles per unit time together with a
sampling frequency :math:`f_s`. For normalized cycles per sample,
:math:`f_s=1`. Whole-band one-sided integrations should report whether the
frequency grid includes the Nyquist endpoint.

The shared ``integrate_spectral_rate`` primitive supports both endpoint-
inclusive trapezoidal integration and the Faes/HOP half-open convention. PIRD's
``half_open=True`` option is specifically intended for the latter.

Reproducible analysis record
----------------------------

A scientific analysis should document the complete transformation

.. math::

   \text{raw data}
   \rightarrow
   \text{preprocessed data}
   \rightarrow
   \text{model/order selection}
   \rightarrow
   \text{fitted model}
   \rightarrow
   \text{diagnostics}
   \rightarrow
   \text{derived measures}.

Data definition
~~~~~~~~~~~~~~~

Report:

- observation unit and channel meaning;
- original and interpreted tensor shapes;
- meaning of the batch dimension;
- whether trajectories are independent;
- trajectory length and count;
- sampling frequency;
- missing-data handling;
- centering, detrending, filtering, downsampling, or normalization;
- any stationarity assumptions.

Repository state
~~~~~~~~~~~~~~~~

Record the exact repository, branch, commit SHA, and package version. Results
from ``main``, an open PR, and an external reference implementation should not
be described as though they were the same software state.

VAR configuration
~~~~~~~~~~~~~~~~~

Report at least

.. math::

   p,\quad
   \text{candidate lags},\quad
   \text{selection method},\quad
   \text{solver},\quad
   \text{covariance convention},\quad
   \text{batch mode}.

Also report effective sample size, spectral radius, innovation covariance, and
failed candidates where applicable.

State-space configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

Report

.. math::

   r,\quad h_p,\quad h_f,

where :math:`r` is latent dimension and :math:`h_p,h_f` are the past and future
block horizons, together with the subspace estimator, ridge regularization,
batch mode, canonical correlations or singular values, selection criterion,
and fitted stability.

Temporal CV configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

Report fold boundaries, gap :math:`g`, prediction mode, gap mode, scoring rule,
selection rule, and whether the selected candidate was refit on all data.

Measure configuration
~~~~~~~~~~~~~~~~~~~~~

For spectral and information-rate analyses, report:

- source, target, conditioning, or group definitions;
- log base;
- frequency grid;
- sampling frequency;
- integration convention;
- exact model representation used.

For O-information rate, report the process grouping explicitly. For PIRD,
report the two or three source groups, target group, whether half-open
integration was used, and whether interpretation refers to raw lattice atoms or
the Faes/HOP coarse-grained unique/redundant/synergistic terms.

For dynamical-dependence optimization, record the optimizer backend, objective
(``proxy`` or ``spectral``), macro dimension, initial projections/restarts,
random seed, convergence codes, and frequency or lag configuration.

Validation identities
---------------------

Where the mathematics provides equivalent temporal and spectral forms, the
identity should be checked numerically. Examples include

.. math::

   F_{Y\to X}
   \approx 2\int_0^{1/2}f_{Y\to X}(\nu)\,d\nu,

spectral O-information-rate integration recovering temporal OIR, and PIRD atom
integration reconstructing the corresponding temporal partial information
rates.

For PIRD, additional useful conservation checks are that Möbius atoms
reconstruct every redundancy function and that integrated atoms reconstruct the
exact source-subset MIR quantities defined by the lattice.

For DARE-dependent measures, parity studies should state which DARE backend was
used and compare against the documented reference backend where appropriate.

Uncertainty and resampling
--------------------------

Model fitting and uncertainty quantification are separate layers. If bootstrap,
permutation, surrogate, or other null procedures are used, report the resampling
unit, null model, number of resamples, random seed, multiple-comparison
correction, and whether model order is reselected inside every resample.

Independent trajectories should remain independent under resampling unless the
scientific null explicitly states otherwise. Random shuffling of individual time
points generally destroys the temporal dependence structure and is not a generic
time-series null model.

Minimal analysis record
-----------------------

A compact record can use the following structure::

   repository: rubenherzog/complextorch
   commit: <exact SHA>
   data shape: (B, T, n)
   batch meaning: independent trajectories
   sampling frequency: <value>
   preprocessing: <explicit steps>

   model: VAR | N4SID | LarimoreStateSpace
   order/dimension: <value>
   selection: <method>
   mode: pooled | independent
   dtype: float64
   device: <device>
   seed: <seed>

   diagnostics:
     spectral radius: <value>
     innovation covariance: <value>
     whiteness/consistency: <value>
     failed candidates: <value>

   measures:
     <explicit list>
   log base: <value>
   frequencies: <definition>
   integration convention: <definition>

Repository references
---------------------

- ``src/complextorch/linalg.py``
- ``src/complextorch/control.py``
- ``src/complextorch/var.py``
- ``src/complextorch/state_space.py``
- ``src/complextorch/selection/``
- ``src/complextorch/measures/``
- ``tests/``
- ``.github/workflows/ci.yml``
