"""
Identify latent state-space structure
=====================================

ComplexTorch provides fixed-dimension N4SID and Larimore CVA estimators. This
example fits both estimators to the same reproducible observations and inspects
their subspace spectra. The state dimension is specified explicitly here;
selection of that dimension is a separate operation in
:mod:`complextorch.selection`.
"""

import matplotlib.pyplot as plt
import torch

from complextorch import LarimoreStateSpace, N4SID, demo_var, simulate_var

# %%
# A VAR process is used only as a convenient known stationary data generator.
# Both state-space estimators receive exactly the same observation trajectories.
coefficients, covariance = demo_var(
    n_variables=3,
    order=2,
    dtype=torch.float64,
)
observations = simulate_var(
    coefficients,
    covariance,
    n_times=1400,
    burnin="auto",
    seed=19,
)

n_states = 4
n4sid = N4SID(
    n_states=n_states,
    block_rows=10,
    mode="pooled",
    dtype="float64",
).fit(observations)

larimore = LarimoreStateSpace(
    n_states=n_states,
    past_horizon=10,
    future_horizon=10,
    mode="pooled",
    dtype="float64",
).fit(observations)

print("N4SID transition shape:", tuple(n4sid.transition_.shape))
print("Larimore transition shape:", tuple(larimore.transition_.shape))
print("Larimore innovation covariance:\n", larimore.innovation_covariance_)

# %%
# The spectra have different meanings: N4SID exposes singular values of its
# future-on-past projection, whereas Larimore exposes canonical correlations of
# the whitened future/past decomposition. They should not be interpreted as the
# same selection criterion.
plt.figure()
plt.semilogy(
    torch.arange(1, n4sid.singular_values_.numel() + 1),
    n4sid.singular_values_.detach().cpu(),
    marker="o",
)
plt.xlabel("subspace component")
plt.ylabel("N4SID singular value")
plt.tight_layout()

plt.figure()
plt.plot(
    torch.arange(1, larimore.canonical_correlations_.numel() + 1),
    larimore.canonical_correlations_.detach().cpu(),
    marker="o",
)
plt.xlabel("canonical component")
plt.ylabel("Larimore canonical correlation")
plt.tight_layout()
