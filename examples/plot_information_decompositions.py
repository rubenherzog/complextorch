"""
Compute model-derived information decompositions
================================================

This example starts from a known VAR model rather than refitting observations.
It converts the process to exact innovations form and evaluates O-information
rate (OIR), partial information rate decomposition (PIRD), and partial
decomposition of Granger causality (PDGC) on a common frequency grid.

PIRD and PDGC use the Faes/HOP convention: redundancy is defined pointwise in
frequency and temporal atoms are obtained by integrating the spectral atoms.
"""

import matplotlib.pyplot as plt
import torch

from complextorch import (
    build_var_system,
    demo_var,
    integrate_spectral_rate,
    o_information_rate,
    partial_granger_causality_decomposition,
    partial_information_rate_decomposition,
    spectral_o_information_rate,
    spectral_partial_granger_causality_decomposition,
    spectral_partial_information_rate_decomposition,
    var_to_innovations_state_space,
)

# %%
# Build a four-variable stationary model. Three variables are treated as source
# groups and the fourth as a target for PIRD/PDGC.
coefficients, covariance = demo_var(
    n_variables=4,
    order=2,
    dtype=torch.float64,
)
var_system = build_var_system(coefficients, covariance)
innovations = var_to_innovations_state_space(var_system)

frequencies = torch.linspace(0.0, 0.5, 129, dtype=torch.float64)
groups = [(0,), (1,), (2,), (3,)]
sources = [0, 1, 2]
target = 3

# %%
# O-information rate can be evaluated directly in time and frequency domains.
temporal_oir = o_information_rate(innovations, groups=groups)
spectral_oir = spectral_o_information_rate(
    innovations,
    frequencies,
    groups=groups,
)
integrated_oir = integrate_spectral_rate(spectral_oir, frequencies)

print(f"temporal OIR: {float(temporal_oir.squeeze()):.6f}")
print(f"integrated spectral OIR: {float(integrated_oir.squeeze()):.6f}")

# %%
# PIRD decomposes source-to-target mutual-information rate. The temporal public
# function performs the spectral decomposition first and only then integrates
# the atoms.
spectral_pird = spectral_partial_information_rate_decomposition(
    innovations,
    sources=sources,
    target=target,
    frequencies=frequencies,
)
temporal_pird = partial_information_rate_decomposition(
    innovations,
    sources=sources,
    target=target,
    frequencies=frequencies,
)
print("PIRD unique rates:", temporal_pird.unique.squeeze(0))
print(f"PIRD redundancy: {float(temporal_pird.redundant.squeeze()):.6f}")
print(f"PIRD synergy: {float(temporal_pird.synergistic.squeeze()):.6f}")

# %%
# PDGC uses the same lattice/coarse-graining infrastructure but decomposes
# unconditional source-subset spectral Granger causality to the target.
spectral_pdgc = spectral_partial_granger_causality_decomposition(
    var_system,
    sources=sources,
    target=target,
    frequencies=frequencies,
)
temporal_pdgc = partial_granger_causality_decomposition(
    var_system,
    sources=sources,
    target=target,
    frequencies=frequencies,
)
print("PDGC unique rates:", temporal_pdgc.unique.squeeze(0))
print(f"PDGC redundancy: {float(temporal_pdgc.redundant.squeeze()):.6f}")
print(f"PDGC synergy: {float(temporal_pdgc.synergistic.squeeze()):.6f}")

# %%
# Compare the frequency-resolved redundancy/synergy balances. These curves are
# not interchangeable: PIRD decomposes mutual-information rate, whereas PDGC
# decomposes directed Granger-causal influence.
plt.figure()
plt.plot(
    frequencies,
    spectral_pird.redundant.squeeze(0),
    label="PIRD redundant",
)
plt.plot(
    frequencies,
    spectral_pird.synergistic.squeeze(0),
    label="PIRD synergistic",
)
plt.xlabel("frequency [cycles/sample]")
plt.ylabel("information-rate density")
plt.legend()
plt.tight_layout()

plt.figure()
plt.plot(
    frequencies,
    spectral_pdgc.redundant.squeeze(0),
    label="PDGC redundant",
)
plt.plot(
    frequencies,
    spectral_pdgc.synergistic.squeeze(0),
    label="PDGC synergistic",
)
plt.xlabel("frequency [cycles/sample]")
plt.ylabel("Granger-causality density")
plt.legend()
plt.tight_layout()
