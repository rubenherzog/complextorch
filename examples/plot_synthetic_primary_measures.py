"""
Synthetic topology grids and primary measures
==============================================

This example compares five three-variable synthetic VAR(1) families on a
20 x 20 parameter grid. The two axes span the admissible range of target
spectral radius ``rho`` and innovation equicorrelation ``r`` while staying just
inside their open stability/positive-definiteness boundaries.

The synthetic families are uncoupled, directed chain, directed ring,
frustrated ring, and modular. All models in each 20 x 20 grid are constructed
and evaluated in one Torch batch. A one-dimensional macro variable is defined
as the leading principal component (PC1) of each stationary covariance for the
emergence and dynamical-dependence measures.

All plotted primary measures are read from one
:func:`complextorch.compute_all_model_measures` result. The example therefore
also exercises the aggregate API contract: shared model primitives and all
applicable primary families are computed by the aggregate entry point rather
than recomputed ad hoc in example code.

The final compact figure uses one column per topology and one row per primary
measure. Color limits are shared across all five topologies within each row so
that colors are directly comparable horizontally.

Abbreviations
-------------

- ``TC``: total correlation.
- ``O``: O-information.
- ``Hdot``: entropy rate.
- ``mean MIR``: mean pairwise mutual-information rate over the three unique
  variable pairs.
- ``max tMVGC``: maximum directed time-domain multivariate Granger causality
  across all ordered singleton source-target pairs.
- ``OIR``: O-information rate.
- ``PI``: predictive information.
- ``Psi``: emergence :math:`\\psi`.
- ``DD``: dynamical dependence.
- ``CA``: covariance amplification.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import torch

from complextorch import ModelMeasureConfig, compute_all_model_measures, synthetic_var

plt.rcParams.update({"font.size": plt.rcParams["font.size"] + 2})


def principal_component_projection(covariance: torch.Tensor) -> torch.Tensor:
    """Return a deterministic one-dimensional PC1 macro-projection."""

    _, eigenvectors = torch.linalg.eigh(covariance)
    pc1 = eigenvectors[..., :, -1]
    signs = torch.where(pc1[..., 0] < 0.0, -1.0, 1.0)
    pc1 = pc1 * signs.unsqueeze(-1)
    return pc1.unsqueeze(-2)


def robust_limits(
    values: torch.Tensor,
    *,
    lower: float = 0.01,
    upper: float = 0.99,
) -> tuple[float, float]:
    """Return row-wise robust color limits from finite values."""

    finite = values[torch.isfinite(values)]
    q = torch.quantile(
        finite,
        torch.tensor([lower, upper], dtype=finite.dtype, device=finite.device),
    )
    vmin = float(q[0])
    vmax = float(q[1])
    if vmax <= vmin:
        vmax = float(finite.max())
        vmin = float(finite.min())
    return vmin, vmax


# %%
# Define the 20 x 20 grid. Both admissible domains are open, so near-extreme
# values are used rather than the singular boundaries themselves.
N_VARIABLES = 3
GRID_SIZE = 20
RHO_VALUES = torch.linspace(0.01, 0.99, GRID_SIZE, dtype=torch.float64)
R_MIN = -1.0 / (N_VARIABLES - 1) + 0.001
R_VALUES = torch.linspace(R_MIN, 0.99, GRID_SIZE, dtype=torch.float64)
RHO_GRID, R_GRID = torch.meshgrid(RHO_VALUES, R_VALUES, indexing="ij")

SYSTEMS: list[tuple[str, str, dict[str, float | int]]] = [
    ("uncoupled", "uncoupled", {}),
    ("directed_chain", "chain", {}),
    ("directed_ring", "ring", {}),
    ("frustrated_ring", "frustrated ring", {}),
    (
        "modular",
        "modular",
        {"n_modules": 2, "within_coupling": 1.0, "between_coupling": 0.15},
    ),
]

MEASURES = [
    ("total_correlation", "TC"),
    ("o_information", "O"),
    ("entropy_rate", "Hdot"),
    ("mean_mir", "mean MIR"),
    ("max_temporal_mvgc", "max tMVGC"),
    ("o_information_rate", "OIR"),
    ("predictive_information", "PI"),
    ("emergence_psi", "Psi"),
    ("dynamical_dependence", "DD"),
    ("covariance_amplification", "CA"),
]

measure_grids: dict[str, dict[str, torch.Tensor]] = {
    key: {} for key, _ in MEASURES
}

for system_name, display_name, parameters in SYSTEMS:
    model = synthetic_var(
        system_name,
        N_VARIABLES,
        spectral_radius_target=RHO_GRID,
        noise_correlation=R_GRID,
        dtype=torch.float64,
        **parameters,
    )
    macro_projection = principal_component_projection(model.present_covariance)
    config = ModelMeasureConfig(
        macro_projection=macro_projection,
        base=2.0,
    )
    measures = compute_all_model_measures(model, config)

    values = {
        "total_correlation": measures["gaussian"]["total_correlation"],
        "o_information": measures["gaussian"]["o_information"],
        "entropy_rate": measures["dynamics"]["entropy_rate"],
        "mean_mir": measures["rates"]["mean_pairwise_mutual_information"],
        "max_temporal_mvgc": measures["mvgc"]["pairwise_temporal"].amax(
            dim=(-2, -1)
        ),
        "o_information_rate": measures["rates"]["o_information"],
        "predictive_information": measures["dynamics"]["predictive_information"],
        "emergence_psi": measures["emergence"]["psi"],
        "dynamical_dependence": measures["control"]["dynamical_dependence"],
        "covariance_amplification": measures["criticality"][
            "covariance_amplification"
        ],
    }
    for key, value in values.items():
        measure_grids[key][display_name] = value.reshape(GRID_SIZE, GRID_SIZE)


# %%
# Plot one row per measure and one column per topology. Every heatmap within a
# row uses the same normalization, and the shared colorbar on the right names
# the corresponding measure using a compact abbreviation.
fig, axes = plt.subplots(
    len(MEASURES),
    len(SYSTEMS),
    figsize=(12.4, 15.2),
    constrained_layout=True,
    squeeze=False,
)
extent = [
    float(R_VALUES.min()),
    float(R_VALUES.max()),
    float(RHO_VALUES.min()),
    float(RHO_VALUES.max()),
]

for row, (measure_name, abbreviation) in enumerate(MEASURES):
    row_values = torch.stack(
        [measure_grids[measure_name][display] for _, display, _ in SYSTEMS], dim=0
    )

    if measure_name == "covariance_amplification":
        positive = row_values[row_values > 0.0]
        q = torch.quantile(
            positive,
            torch.tensor([0.01, 0.98], dtype=positive.dtype, device=positive.device),
        )
        vmin = max(float(q[0]), torch.finfo(positive.dtype).tiny)
        vmax = max(float(q[1]), vmin * 1.01)
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        cmap = "viridis"
    else:
        vmin, vmax = robust_limits(row_values)
        if vmin < 0.0 < vmax and measure_name in {
            "o_information",
            "o_information_rate",
            "emergence_psi",
        }:
            bound = max(abs(vmin), abs(vmax))
            norm = mcolors.TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
            cmap = "coolwarm"
        else:
            if vmax <= vmin:
                vmax = vmin + 1e-12
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cmap = "viridis"

    images = []
    for col, (_, display_name, _) in enumerate(SYSTEMS):
        axis = axes[row, col]
        image = axis.imshow(
            measure_grids[measure_name][display_name],
            origin="lower",
            extent=extent,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
        )
        images.append(image)

        if row == 0:
            axis.set_title(display_name)
        if row == len(MEASURES) - 1:
            axis.set_xlabel("r")
        else:
            axis.set_xticklabels([])
        if col == 0:
            axis.set_ylabel("rho")
        else:
            axis.set_yticklabels([])

    colorbar = fig.colorbar(
        images[-1],
        ax=axes[row, :],
        shrink=0.9,
        pad=0.015,
        aspect=18,
    )
    colorbar.set_label(abbreviation, rotation=270, labelpad=14)

plt.show()
