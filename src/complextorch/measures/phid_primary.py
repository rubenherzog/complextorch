"""Model-first PhiID redundancy entry point.

This adapter reuses the canonical model-autocovariance backbone. It contains no
estimator and never derives covariance from observations.
"""
from __future__ import annotations

import torch

from .phid import PhiIDRedundancy, gaussian_phiid_atoms
from .primary import CovarianceModel, past_future_covariance


def phiid_redundancy_from_model(
    model: CovarianceModel,
    variables: tuple[int, int],
    *,
    lag: int = 1,
    redundancy: PhiIDRedundancy | str = "mmi",
    base: float = 2.0,
    ccs_qmc_samples: int = 4096,
    autocovariance_sequence: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute bivariate PhiID from a supplied Gaussian generative model.

    Parameters
    ----------
    model
        Canonical :class:`~complextorch.representations.VARSystem` or
        :class:`~complextorch.representations.StateSpaceModel`.
    variables
        Two distinct observed-variable indices defining the bivariate process.
    lag
        Positive past-to-future lag in samples.
    redundancy
        ``"mmi"``, ``"ccs"``, ``"idep_a"``, or ``"idep_b"``.
    base
        Logarithm base for all information quantities.
    ccs_qmc_samples
        Deterministic Sobol integration-node count used only by Gaussian CCS.
    autocovariance_sequence
        Optional model-derived autocovariance sequence from a shared
        :class:`~complextorch.measures.primary.ModelMeasureContext`.

    Returns
    -------
    dict[str, torch.Tensor]
        Sixteen PhiID atoms plus ``total`` and ``reconstruction``. Leading model
        batch dimensions are preserved.

    Notes
    -----
    The only numerical integration used here is the expectation defining
    Gaussian CCS. Its Sobol nodes are drawn from the supplied model covariance;
    they are not observations and no parameter is refitted.
    """
    covariance = past_future_covariance(
        model,
        variables,
        lag=lag,
        autocovariance_sequence=autocovariance_sequence,
    )
    return gaussian_phiid_atoms(
        covariance,
        redundancy=redundancy,
        base=base,
        ccs_qmc_samples=ccs_qmc_samples,
    )


__all__ = ["phiid_redundancy_from_model"]
