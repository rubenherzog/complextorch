"""Model-first Gaussian entropy-rate extensions.

This module provides frequency-resolved and exact marginal entropy rates for
stationary linear-Gaussian models.  Both quantities are derived from canonical
innovations representations; no observations are refit.

References
----------
- Cover, T. M. and Thomas, J. A. (2006). *Elements of Information Theory*.
- Barnett, L. and Seth, A. K. (2015). Granger causality for state-space models.
- Mediano, P. et al. EntRate ``StateSpaceEntropyRate`` implementation.
"""
from __future__ import annotations

import math

import torch

from ..control import reduce_innovations_state_space
from ..spectra import hermitian_logdet, innovations_spectral_density
from .backbone import Model, as_innovations
from .gaussian import gaussian_entropy


def marginal_entropy_rate(
    model: Model,
    *,
    base: float = 2.0,
) -> torch.Tensor:
    r"""Return one exact Gaussian entropy rate per observed variable.

    Each channel is marginalised from the full multivariate innovations model
    through the canonical generalized-DARE reduction before its entropy rate is
    evaluated.  This is not the entropy of a diagonal block of the microscopic
    innovations covariance: marginalisation changes the innovations process.

    For channel :math:`i` with exact marginal innovations variance
    :math:`V_i^R`,

    .. math::

       h_i = \frac12\log_b\left(2\pi e\,V_i^R\right).

    Parameters
    ----------
    model
        ``VARSystem``, ``StateSpaceModel``, or ``InnovationsStateSpace``.
    base
        Logarithm base used for the differential entropy rate.

    Returns
    -------
    torch.Tensor
        Shape ``(n_variables,)`` for an unbatched model or
        ``(batch, n_variables)`` for a batched model.  No averaging across
        variables is performed.
    """
    innovations = as_innovations(model)
    n_observations = innovations.observation.shape[-2]
    values = []
    for index in range(n_observations):
        marginal = reduce_innovations_state_space(innovations, (index,))
        values.append(gaussian_entropy(marginal.innovation_covariance, base=base))
    return torch.stack(values, dim=-1)


def spectral_entropy_rate_from_spectrum(
    spectrum: torch.Tensor,
    *,
    sampling_frequency: float = 1.0,
    base: float = 2.0,
) -> torch.Tensor:
    r"""Return Gaussian entropy-rate density from a cross-spectral density.

    ComplexTorch spectral densities are expressed per unit frequency,
    ``S(f) = H(f) V H(f)^* / Fs``.  The frequency-resolved entropy-rate density
    is therefore

    .. math::

       h(f)=\frac12\log_b\det\left(2\pi e\,F_s S(f)\right).

    The factor ``Fs`` removes the arbitrary per-Hz density scaling.  On a full
    one-sided real-process grid, applying ComplexTorch's standard
    ``integrate_spectral_rate`` convention to ``h(f)`` recovers the broadband
    entropy rate.  This function performs no integration and no band averaging.

    Parameters
    ----------
    spectrum
        Hermitian positive-definite cross-spectral density with shape
        ``(..., frequencies, n_variables, n_variables)``.
    sampling_frequency
        Positive sampling frequency associated with the spectral density.
    base
        Logarithm base used for the differential entropy-rate density.

    Returns
    -------
    torch.Tensor
        Entropy-rate density with shape ``(..., frequencies)``.
    """
    value = torch.as_tensor(spectrum)
    if value.ndim < 3 or value.shape[-2] != value.shape[-1]:
        raise ValueError(
            "spectrum must have shape (..., frequencies, n_variables, n_variables)"
        )
    if not math.isfinite(float(sampling_frequency)) or sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be finite and positive")
    if not math.isfinite(float(base)) or base <= 0 or base == 1:
        raise ValueError("base must be finite, positive, and different from one")

    n_variables = value.shape[-1]
    logdet = hermitian_logdet(value)
    constant = n_variables * math.log(
        2.0 * math.pi * math.e * float(sampling_frequency)
    )
    return 0.5 * (constant + logdet) / math.log(float(base))


def spectral_entropy_rate(
    model: Model,
    frequencies: torch.Tensor,
    *,
    sampling_frequency: float = 1.0,
    base: float = 2.0,
) -> torch.Tensor:
    r"""Return frequency-resolved Gaussian entropy rate for a model.

    The model is converted to canonical innovations form, its exact spectral
    density is evaluated on the user-supplied frequency grid, and
    :func:`spectral_entropy_rate_from_spectrum` is applied.  Frequencies are in
    cycles per unit time and must lie in ``[0, sampling_frequency/2]``.

    No integration across frequencies or user-defined bands is performed.
    """
    innovations = as_innovations(model)
    spectrum = innovations_spectral_density(
        innovations,
        frequencies,
        sampling_frequency=sampling_frequency,
    )
    return spectral_entropy_rate_from_spectrum(
        spectrum,
        sampling_frequency=sampling_frequency,
        base=base,
    )


__all__ = [
    "marginal_entropy_rate",
    "spectral_entropy_rate",
    "spectral_entropy_rate_from_spectrum",
]
