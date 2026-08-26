# Architecture

ComplexTorch separates six layers:

1. Adapters normalize external layouts to `(batch, time, variables)`.
2. Estimators infer VAR or state-space parameters independently of downstream measures.
3. Canonical representations (`VARSystem`, `StateSpaceModel`, and `InnovationsStateSpace`) preserve model semantics and provide exact conversion paths.
4. Selection evaluates estimators with trajectory-aware temporal folds.
5. Measures consume canonical representations rather than estimator internals.
6. Mechanistic/design utilities act on canonical model invariants or explicit design parameters without refitting observations.

The core data flow is:

```text
raw trajectories -> estimator -> canonical dynamical system -> measure planner -> outputs
                                      |
                                      +-> modal mechanisms / prescribed design
```

Independent trajectories use `(batch, time, variables)` and remain independent throughout fitting, validation, lag construction, and resampling. No transition, lag, residual pair, or validation link is constructed across trajectory boundaries.

`VARSystem` is the canonical fitted autoregressive process. `StateSpaceModel` represents a general latent linear Gaussian system, while `InnovationsStateSpace` represents its steady-state innovations form. These objects are not interchangeable: the innovations form is a process representation used for exact spectral, entropy-rate, Granger-causality, emergence, and projection calculations.

Model transformations and mechanistic calculations operate on the common innovations representation when possible. The pole--residue layer exposes similarity-invariant transfer-function modes for simple diagonalizable transitions. The design layer does not introduce a new model class: a user supplies continuous parameters and a batched mapping from those parameters to existing ComplexTorch capabilities.

All numerical paths preserve dtype/device and use Torch batched linear algebra where practical. Linear systems and factorizations are preferred to explicit matrix inverses. Float64 remains the recommended dtype for fitting, Riccati/Lyapunov calculations, and analytical validation.
