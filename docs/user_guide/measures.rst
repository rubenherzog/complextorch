Measures
========

ComplexTorch distinguishes model-derived analytical measures from observation
estimators. The model-derived layer consumes a supplied
:class:`~complextorch.VARSystem`, :class:`~complextorch.StateSpaceModel`, or
:class:`~complextorch.InnovationsStateSpace` and does not silently refit
observations.

The measure documentation is organized by scientific object rather than by
implementation module. This keeps static Gaussian information, dynamical
information, directed dependence, high-order dynamics, emergence, criticality,
and control-theoretic quantities conceptually separate.

.. toctree::
   :maxdepth: 3

   measures/gaussian_information
   measures/dynamical_information
   measures/directed_information
   measures/high_order/index
   measures/emergence/index
   measures/criticality

Control-theoretic quantities are documented separately in :doc:`control`.

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
- ``src/complextorch/dd_ssdi.py``
