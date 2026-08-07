# SSDI / State-Space Validation Report 002

Branch: `ssdi-testing`

## Scope of this stage

This report records the lightweight validation layer for the collaborator-requested
state-space dynamical-dependence study. The heavy experiment and publication
figures are intentionally deferred until these contracts are executable and clean.

Clarified experiment specification:

- microscopic observation dimension: `n = 8`;
- macro dimensions: every nontrivial dimension `m = 1, ..., 7`;
- optimization runs/restarts: the same concept, with 100 independent random
  Grassmann initializations per macro dimension in the heavy experiment;
- optimizer ceiling: 10,000 gradient iterations per restart;
- projection storage: ComplexTorch-native `(runs, m, n)` only;
- Figure 4 supplied by the collaborator is a visualization reference for a later
  stage, not part of this validation layer.

## Fixed 8-node starting system

`scripts/ssdi_macro_study.py` introduces `fixed_modular_8_mask()` and
`fixed_modular_8_system()`.

The state-matrix convention is `A[target, source]`, so a nonzero matrix entry is
an edge `source -> target`.

The predefined topology is:

- nodes 0--1: internally fully connected 2-node module;
- nodes 2--7: internally fully connected 6-node module;
- exactly one cross-module edge: node 1 -> node 2;
- no edge from the 6-node module back to the 2-node module.

Nonzero transition weights are generated from a pinned seed and the resulting
matrix is rescaled to spectral radius 0.72, matching the stability convention
used by the existing SSDI validation systems.

## Macro sweep contract

`run_macro_dimension_sweep(...)` is a reusable lightweight driver over the
existing public `optimise_dynamical_dependence(...)` API.

For each `m = 1, ..., n-1` it:

1. generates an independent random row-orthonormal restart tensor with shape
   `(restarts, m, n)`;
2. calls the selected existing optimizer without changing its API;
3. retains the native `DDOptimizationResult`;
4. optionally retains optimization history for the later DD-vs-iteration plot.

The heavy study will call this with `restarts=100`, `max_iterations=10000`, and
`history=True`. Those values are not executed in CI.

Both current ComplexTorch optimizer backends are compatible with this storage
contract: `complexbox` and `riemannian_armijo`.

## Subspace metric / collaborator "similarity matrix"

The collaborator referred to a 100 x 100 similarity matrix and to MATLAB
`subspacea.m` plus `gmetric*.m`. The pinned ComplexBox implementation confirms
that the reference quantity is actually a normalized Grassmann **distance**:

- principal angles are obtained by `subspacea`;
- default `gmetric` is the largest principal angle divided by `pi/2`;
- `gmetrics` constructs the symmetric pairwise run-by-run matrix.

The new `grassmann_distance_matrix(...)` reproduces this convention directly in
ComplexTorch row orientation and evaluates all run-pair cross-Gram matrices with
batched Torch SVD. For 100 optimized projections its output shape is `(100,100)`.

No MATLAB-style `(n,m,runs)` representation is stored internally. Transposition
is used only inside optional external-parity tests at the ComplexBox boundary.

## Micro-to-macro loading contract

Raw entries of an optimized basis are not identifiable because rotating the
basis inside a fixed macro subspace changes those entries without changing the
subspace. Therefore the validation layer uses basis-invariant quantities from
the same SSDI/ComplexBox geometry family:

- `micro_macro_loadings(M) = sum_j M[j,i]^2`, equivalent to ComplexBox
  `habeta(L)` under `L=M.T`;
- `coordinate_axis_distances(M)`, equivalent to ComplexBox `gmetrics1(L)`.

For an `m`-dimensional macro subspace, the squared loadings lie in `[0,1]` and
sum exactly to `m`. These are the quantities intended for the later
micro-to-macro loading analysis.

## Added tests

`tests/test_ssdi_macro_study.py` adds seven lightweight contracts:

1. exact 8-node 2+6 connectivity and single directed bridge;
2. stable SSM and macro dimensions exactly 1--7;
3. full macro sweep with ComplexBox-compatible optimizer, including native
   projection orientation, sorted final DD objective, orthonormality, and stored
   histories;
4. the Riemannian Armijo backend obeys the same storage/history contract;
5. the intended 100-run geometry produces a symmetric 100 x 100 normalized
   distance matrix with zero diagonal;
6. micro-to-macro loadings and coordinate-axis distances are invariant to basis
   rotation and satisfy their mathematical normalization identities;
7. when pinned ComplexBox is installed, `gmetrics`, `gmetrics1`, and `habeta`
   parity is checked numerically.

The existing `tests/test_ssdi_validation.py` remains unchanged and continues to
cover equation-level DD/DARE/gradient/parity behavior.

## Figure 4 reference

The supplied Figure 4 shows the qualitative target style for a later analysis:
separate macro-scale panels, highlighted dominant structure, and network insets.
It corresponds to a different 9-node example. This branch's requested starting
system is explicitly 8-node, so the future analogous sweep is `m=1,...,7`.
No attempt to reproduce the figure is made before the numerical study passes.

## Validation status

Implementation and diff inspection are complete for this layer, but executable
validation has not yet been claimed.

The current sandbox cannot materialize the repository checkout through its
network. The GitHub connector remains available for repository reads/writes.
A PR has not been created, so the permanent PR-triggered GitHub Actions workflow
has not been invoked for this branch.

Before beginning the heavy layer, the required gate is:

- focused tests: `tests/test_ssdi_macro_study.py` and `tests/test_ssdi_validation.py`;
- complete repository pytest suite;
- GitHub Actions Python 3.10 success;
- GitHub Actions Python 3.12 success;
- exact validation head SHA recorded.

CUDA is separate and is not required for correctness of this CPU validation
layer.
