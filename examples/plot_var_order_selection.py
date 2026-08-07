"""
Select VAR lag order with information criteria
==============================================

Model selection is separate from fixed-order fitting in ComplexTorch. This
example simulates a VAR(2), evaluates independently fitted candidate orders with
AIC, BIC, and HQC, and then uses the HQC-selected estimator exposed by
:class:`complextorch.VAROrderSelectionIC`.
"""

import matplotlib.pyplot as plt
import torch

from complextorch import VAROrderSelectionIC, demo_var, simulate_var

# %%
# Simulate a known second-order system. A longer record makes the information
# criterion curves easier to interpret while remaining inexpensive to execute.
coefficients, covariance = demo_var(
    n_variables=3,
    order=2,
    dtype=torch.float64,
)
observations = simulate_var(
    coefficients,
    covariance,
    n_times=1800,
    burnin="auto",
    seed=11,
)

selector = VAROrderSelectionIC(
    orders=range(1, 7),
    solver="lwr",
    refit="hqc",
    dtype="float64",
).fit(observations)

print(f"AIC order: {selector.p_aic_}")
print(f"BIC order: {selector.p_bic_}")
print(f"HQC order: {selector.p_hqc_}")
print(f"refitted order: {selector.best_order_}")

# %%
# The complete criterion curves should be retained in scientific analyses, not
# only the minimizing order.
orders = selector.result_.orders
plt.figure()
plt.plot(orders, selector.aic_, marker="o", label="AIC")
plt.plot(orders, selector.bic_, marker="o", label="BIC")
plt.plot(orders, selector.hqc_, marker="o", label="HQC")
plt.xlabel("VAR lag order")
plt.ylabel("criterion")
plt.xticks(orders)
plt.legend()
plt.tight_layout()
