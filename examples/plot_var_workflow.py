"""
Fit and diagnose a VAR model
============================

This example simulates a known stable VAR(2), fits it with the public
:class:`complextorch.VAR` estimator, converts the estimate to the canonical
:class:`complextorch.VARSystem`, and compares one-step predictions with the
observations.

The synthetic generator uses a fixed seed. The simulated data have shape
``(batch, time, variables)`` and the fit uses ``mode="pooled"``; lagged rows are
constructed inside each trajectory before pooling.
"""

import matplotlib.pyplot as plt
import torch

from complextorch import VAR, demo_var, simulate_var

# %%
# Generate a reproducible stable process.
coefficients, innovation_covariance = demo_var(
    n_variables=3,
    order=2,
    dtype=torch.float64,
)
observations = simulate_var(
    coefficients,
    innovation_covariance,
    n_times=1200,
    burnin="auto",
    seed=7,
)

# %%
# Fit the same lag order. ``fit`` returns the estimator itself and learned
# quantities use scikit-learn-style trailing underscores.
estimator = VAR(
    order=2,
    solver="lstsq",
    mode="pooled",
    covariance="mle",
    dtype="float64",
).fit(observations)
system = estimator.to_var_system()

predictions = estimator.one_step_predictions(observations)
targets = observations[:, estimator.order :, :]
rmse = torch.sqrt(torch.mean((targets - predictions) ** 2))

print(f"RMSE: {float(rmse):.4f}")
print(f"spectral radius: {float(system.spectral_radius.max()):.4f}")
print("innovation covariance:\n", estimator.noise_covariance_[0])

# %%
# Plot one variable over a short interval. The first prediction corresponds to
# time index ``order`` because the first two samples initialize the VAR history.
window = 180
x_axis = torch.arange(estimator.order, estimator.order + window)

plt.figure()
plt.plot(x_axis, targets[0, :window, 0], label="observed")
plt.plot(x_axis, predictions[0, :window, 0], label="one-step prediction")
plt.xlabel("time sample")
plt.ylabel("variable 0")
plt.legend()
plt.tight_layout()
