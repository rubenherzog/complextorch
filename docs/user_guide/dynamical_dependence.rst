Dynamical independence and SSDI optimization
============================================

``optimise_dynamical_dependence`` defaults to the complete Barnett--Seth SSDI
workflow: innovation whitening, many proxy-DD pre-optimizations, Grassmann
``gmetrics`` distances, greedy ``Lcluster``, spectral-DD refinement of one
representative per cluster, and inverse transformation to physical observation
coordinates.

For a VAR(p) with innovation covariance ``V = B B.T``, the proxy stage uses
``B^{-1} A_k B`` directly for every VAR coefficient ``A_k``, matching SSDI
``cak2ddx``. The spectral stage uses the exactly equivalent innovations
state-space transfer function. For state-space input, the existing innovations
``C A^(k-1) K`` proxy convention is retained.

The default numerical backend is ``optimizer="complexbox"``. The optional
``optimizer="riemannian_armijo"`` executes the same scientific workflow with a
different native-Torch search. Explicit ``objective="proxy"`` or
``objective="spectral"`` calls retain the previous one-stage behavior.

The staged reference-compatible search uses variant 1 by default. Reference
variant 2 moves the current projection even when its best-so-far scalar
objective is not updated, so its returned endpoint can cease to correspond to
the reported scalar. Variant 2 remains available explicitly for literal
reference-behavior studies.

Default staged settings are 100 proxy restarts, a 10,000-iteration ceiling per
stage, ``cluster_tolerance=1e-6``, and 513 equally spaced one-sided normalized
frequency points unless a grid is supplied. The returned result retains the
complete proxy result, cluster representative indices, cluster sizes, and the
pairwise Grassmann distance matrix.

Validation includes a nested VAR benchmark whose structurally closed
macrospaces have dimensions 2 and 6, plus parity tests for the Grassmann metric,
clustering, covariance/subspace transformations, and DD kernels.

References
----------

- Barnett and Seth (2023), Physical Review E 108, 014304.
- MATLAB SSDI, ``lcbarnett/ssdi`` commit
  ``b38ce65f9df18916da216848560c1789e456c04f``.
- ComplexBox, ``bmilinkovic/complexbox`` commit
  ``87b5e2cd9bba22ddd978bade6f614da7d6190db2``.
