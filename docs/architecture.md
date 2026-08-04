# Architecture

ComplexTorch separates five layers:

1. Adapters normalize external layouts to `(batch, time, variables)`.
2. Estimators infer VAR or future SSM parameters independently of downstream measures.
3. Representations convert fitted models to a common `LinearDynamicalSystem` or `VARSystem`.
4. Selection evaluates estimators with epoch-aware temporal folds.
5. Measures consume canonical representations rather than estimator internals.

The data flow is:

```text
raw epochs -> estimator -> canonical dynamical system -> measure planner -> outputs
```

The first implementation focuses on VAR because it is identifiable in observed coordinates, has closed-form OLS/Ridge estimation, converts exactly to companion state space, and permits numerical validation against established implementations. General latent SSM identification is planned separately.

All batched operations use the leading dimension for independent epochs or systems. Float64 is the default for fitting and analytical measures. OLS uses `torch.linalg.lstsq`; Ridge uses Cholesky solves; stationary covariance uses a batched discrete Lyapunov solver; stability is defined by the companion eigenvalue radius.
