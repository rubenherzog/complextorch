"""
Pool independent trajectories without crossing boundaries
=========================================================

A leading batch dimension in ComplexTorch represents independent trajectories,
not one long concatenated time series. This example simulates four independent
realizations of the same VAR(2) and contrasts ``mode="pooled"`` with
``mode="independent"``.

Pooled fitting estimates one shared parameter set, but lagged regression rows
are constructed within each trajectory before pooling. Independent fitting
returns one parameter set per trajectory.
"""

import matplotlib.pyplot as plt
import torch

from complextorch import VAR, demo_var, simulate_var

# %%
# Replicate the same generative model over four independent trajectories. The
# simulator draws independent innovations for every leading-batch element.
coefficients, covariance = demo_var(
    n_variables=3,
    order=2,
    dtype=torch.float64,
)
coefficients = coefficients.expand(4, -1, -1, -1).clone()
covariance = covariance.expand(4, -1, -1).clone()
observations = simulate_var(
    coefficients,
    covariance,
    n_times=700,
    burnin="auto",
    seed=23,
)

pooled = VAR(
    order=2,
    mode="pooled",
    solver="lstsq",
    dtype="float64",
).fit(observations)
independent = VAR(
    order=2,
    mode="independent",
    solver="lstsq",
    dtype="float64",
).fit(observations)

print("observations shape:", tuple(observations.shape))
print("pooled coefficients shape:", tuple(pooled.coef_.shape))
print("independent coefficients shape:", tuple(independent.coef_.shape))

# %%
# Compare coefficient errors. The pooled estimate uses all valid trajectory-local
# rows and therefore often has lower sampling variance when the common-model
# assumption is correct.
truth = coefficients[0]
pooled_error = torch.linalg.vector_norm(pooled.coef_[0] - truth)
independent_error = torch.linalg.vector_norm(
    independent.coef_ - truth.unsqueeze(0), dim=(1, 2, 3)
)

print(f"pooled coefficient error: {float(pooled_error):.4f}")
print("independent coefficient errors:", independent_error)

plt.figure()
plt.bar(
    ["pooled", "trial 0", "trial 1", "trial 2", "trial 3"],
    torch.cat((pooled_error.reshape(1), independent_error)).detach().cpu(),
)
plt.ylabel("coefficient Frobenius error")
plt.xticks(rotation=25)
plt.tight_layout()
