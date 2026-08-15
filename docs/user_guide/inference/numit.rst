NuMIT null-model normalisation
==============================

:func:`~complextorch.numit_pid_var` implements the VAR form of Null Models for
Information Theory (NuMIT) introduced by Liardi et al. (2025). NuMIT is designed
for comparing PID structure across systems whose raw PID atoms may differ simply
because their total mutual information (TMI) differs.

Scientific idea
---------------

For an observed two-source PID, NuMIT compares each PID atom with an ensemble of
otherwise-random systems constrained to have the same TMI. The normalized value
is the empirical quantile of the observed atom in the corresponding null
ensemble. Thus the result is a position relative to a TMI-matched reference
distribution, not a confidence interval around the observed estimate.

For the Gaussian VAR implementation, ComplexTorch uses

.. math::

   I_{\mathrm{TMI}}
   = I\!\left(X_t; X_{t-1:t-p}\right),

which is the stationary past--future mutual information returned by
:func:`~complextorch.var_total_mutual_information`.

VAR null construction
---------------------

The implemented procedure follows the VAR construction in Liardi et al.:

1. Compute the observed Gaussian PID and its TMI.
2. Draw random VAR coefficient shapes and random innovation covariance matrices.
3. For every null VAR, tune the companion spectral radius until its TMI matches
   the observed TMI.
4. Compute the same PID atoms on the TMI-matched null ensemble.
5. Report the empirical mid-quantile of each observed atom with respect to its
   null distribution.

The spectral-radius matching is performed in batch with a stable bisection
procedure. The result object also retains the matched null TMI values and null
spectral radii so the quality of the constrained ensemble can be inspected.

Interpretation
--------------

A NuMIT quantile near 0.5 places an observed atom near the centre of the
TMI-matched null distribution. Values approaching 0 or 1 place it toward the
lower or upper tail, respectively. These quantiles support comparisons of PID
structure after conditioning on TMI; they should not be interpreted as
bootstrap confidence levels.

Current public contract
-----------------------

The current VAR API accepts one observed :class:`~complextorch.VARSystem` and a
disjoint, exhaustive, equal-sized partition ``source0`` / ``source1``. The PID
redundancy definition is selected with ``redundancy``. ``n_null`` controls the
null-ensemble size and ``seed`` provides reproducible local Torch sampling.

For MMI PID, the implementation follows the reference convention for the unique
information quantile by pooling the two unique-null contributions before
computing the corresponding empirical quantiles.

Main API
--------

- :func:`~complextorch.numit_pid_var`
- :func:`~complextorch.var_total_mutual_information`
- :class:`~complextorch.NuMITPIDResult`

Reference and parity target
---------------------------

Liardi, A., Rosas, F. E., Carhart-Harris, R. L., Blackburne, G., Bor, D., &
Mediano, P. A. M. (2025). *Null models for comparing information decomposition
across complex systems*. PLOS Computational Biology, 21(11), e1013629.
https://doi.org/10.1371/journal.pcbi.1013629

The ComplexTorch VAR implementation was validated against the authors' public
``alberto-liardi/NuMIT`` reference implementation at commit
``44efc720c963afb011d376aa9682006657f8c3c0``.
