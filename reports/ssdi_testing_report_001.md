# SSDI / State-Space Validation Report 001

Branch: `ssdi-testing`  
ComplexTorch baseline: `a6bbb645f47771ef0a20ed9f4baf2583b5c259db`  
ComplexBox reference: `bmilinkovic/complexbox@87b5e2cd9bba22ddd978bade6f614da7d6190db2`  
MVGC2 reference: `lcbarnett/MVGC2@b22d3f0f061dcc40ba0e6cb31e636feb3183436d`

## Purpose

This branch is dedicated to scientific validation of state-space dynamical
dependence (SSDI/DD), not to production API changes. The study tests:

1. innovations-form state-space equations used by DD;
2. proxy and spectral DD equations and analytic gradients;
3. the two ComplexTorch optimizers currently exposed by the unified API:
   `complexbox` and `riemannian_armijo`;
4. Grassmann/subspace geometry through principal angles;
5. parity with ComplexBox where the external package is installed;
6. structured synthetic systems with known topology and one analytically planted
   macro-subspace;
7. wall time, numerical stability, convergence, and local-minimum structure.

## Reference conventions

ComplexTorch stores a macro projection as a row-orthonormal matrix `M` with
shape `(m, n)`. ComplexBox/SSDI stores the same subspace by columns
`L = M.T` with shape `(n, m)`.

For identity innovations covariance, the proxy sequence is

\[
Q_k = C A^{k-1} K,
\]

and the proxy objective is

\[
D_x(M)=\sum_k \|M Q_k\|_F^2
       -\|M Q_k M^T\|_F^2.
\]

For a projected innovations model, the native equation-level test constructs

\[
C_R=MC,\quad Q=K V K^T,\quad R=M V M^T,\quad S=K V M^T,
\]

solves the generalized filtering DARE with both the SciPy and Torch backends,
and independently evaluates its algebraic residual and resulting projected
innovation log-determinant. This is compared with the public exact-DD function.

Exact DD is compared in natural-log units (`base=e`) because that is the
ComplexBox/SSDI convention.

Subspace comparisons use principal angles. For orthonormal column
representatives `Q_a` and `Q_b`, the singular values of `Q_a^T Q_b` are the
cosines of the principal angles. The reported Grassmann distance is the
Euclidean norm of those angles.

## Synthetic systems

The executable study is `scripts/validate_ssdi.py`.

It includes these deterministic cases:

- `tnet5`: exact binary connectivity mask from MVGC2 `demo/tnet5.m`;
- `tnet9`: exact binary connectivity mask from MVGC2 `demo/tnet9.m`;
- `erdos_renyi_sparse`: directed Erdos--Renyi graph;
- `erdos_renyi_dense`: directed Erdos--Renyi graph;
- `modular`: directed stochastic-block graph with high within-module and low
  between-module connection probability;
- `random_network`: unconstrained directed random graph;
- `planted_modular`: independent symmetric two-node modules whose normalized
  module averages form an invariant macro-dynamical subspace.

Random transition weights are rescaled to spectral radius `0.72`. This keeps
all graph-derived systems strictly stable and prevents stability differences
from confounding optimizer comparisons.

The planted modular system is the ground-truth structural case. Its macrospace
is explicitly known before optimization. Exact DD at that projection is tested
to be numerically zero, and optimizer endpoints are compared with the planted
subspace using principal-angle distance.

## Test inventory

`tests/test_ssdi_validation.py` contains 20 collected tests after parameter
expansion:

- 1 exact MVGC2 mask test;
- 6 stability/shape tests, one for each graph-derived system;
- 1 planted closed-macrospace exact-DD test;
- 1 direct state-space recurrence test for `C A^(k-1) K`;
- 1 independent-loop proxy-equation test;
- 1 finite-difference test of the analytic proxy Grassmann gradient;
- 1 principal-angle basis-invariance test;
- 1 local-minimum clustering basis-invariance test;
- 2 optimizer tests (`complexbox`, `riemannian_armijo`) for finite values,
  orthonormal endpoints, and objective improvement;
- 2 generalized-DARE equation/residual tests (`scipy`, `torch`);
- 1 ComplexBox proxy + exact-DD parity test;
- 1 ComplexBox optimizer endpoint parity test;
- 1 ComplexBox spectral objective + gradient parity test.

The three ComplexBox tests use `pytest.importorskip("complexbox")`. Therefore the
normal ComplexTorch development environment can still run the complete native
suite without acquiring an undeclared production dependency, while an
environment with the pinned ComplexBox checkout executes the external parity
layer.

## Benchmark outputs

For each system and optimizer, the executable reports:

- wall-clock seconds;
- best, mean, and standard deviation of final proxy objective;
- best exact DD evaluated by the canonical generalized-DARE path;
- finite-result rate;
- convergence rate;
- maximum row-orthonormality error;
- number of distinct local-minimum clusters;
- size of the largest local-minimum cluster;
- median pairwise Grassmann distance among endpoints;
- nearest distance to the planted macrospace when ground truth exists.

When ComplexBox is installed and the study runs on CPU, the same initial
Grassmann subspaces are supplied to ComplexBox. The report additionally records:

- maximum absolute objective discrepancy for the ComplexBox-compatible
  ComplexTorch optimizer;
- nearest-neighbour endpoint distances between toolboxes in both directions;
- best matched subspace distance.

This avoids comparing raw projection matrices, which are non-identifiable up to
an orthogonal change of basis within a subspace.

## Recommended execution

Native ComplexTorch validation:

```bash
pytest -q tests/test_ssdi_validation.py
python scripts/validate_ssdi.py --runs 32 --lags 16 --max-iterations 500 \
    --output ssdi_validation_cpu.json
```

External parity environment:

```bash
# Install ComplexBox from the exact pinned commit in an isolated validation env.
pytest -q tests/test_ssdi_validation.py
python scripts/validate_ssdi.py --runs 32 --lags 16 --max-iterations 500 \
    --output ssdi_validation_complexbox.json
```

GPU benchmark, where available:

```bash
python scripts/validate_ssdi.py --device cuda --runs 128 --lags 32 \
    --max-iterations 500 --output ssdi_validation_cuda.json
```

ComplexBox comparison is intentionally CPU-only because its NumPy reference path
is the scientific baseline; the GPU study compares the two ComplexTorch
optimizers only.

## Validation status

Repository inspection and branch creation are complete.

Local execution is currently unavailable because the sandbox cannot resolve
`github.com` for a checkout. This is a sandbox-network limitation, not a GitHub
connector failure. No numerical pass/fail claim is made in this report until
the branch is executed either locally or through the repository's permanent CI.

No PR has been created and no merge is implied by this report.
