"""Shared spectral primitives for stationary linear Gaussian processes.

This module centralises the Hermitian spectral operations used by information
rates and future multiscale measures.  Frequencies are expressed in cycles per
unit time, matching the existing ComplexTorch convention.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .control import InnovationsStateSpace, innovations_transfer_function


def hermitian_part(matrix: torch.Tensor) -> torch.Tensor:
    """Return the Hermitian part ``(X + X*) / 2`` of a matrix or matrix batch."""
    value = torch.as_tensor(matrix)
    return 0.5 * (value + value.conj().transpose(-1, -2))


def hermitian_logdet(matrix: torch.Tensor) -> torch.Tensor:
    r"""Return ``log(det(X))`` for Hermitian positive-definite matrices.

    A Cholesky factorisation is used instead of a generic complex determinant,
    so positivity is validated and the result is real-valued:

    .. math::

       \log |X| = 2\sum_i \log L_{ii},\qquad X=LL^*.
    """
    value = hermitian_part(torch.as_tensor(matrix))
    factor = torch.linalg.cholesky(value)
    diagonal = torch.diagonal(factor, dim1=-2, dim2=-1).real
    return 2.0 * torch.log(diagonal).sum(dim=-1)


def innovations_spectral_density(
    system: InnovationsStateSpace,
    frequencies: torch.Tensor,
    *,
    sampling_frequency: float = 1.0,
) -> torch.Tensor:
    r"""Return the observation spectral-density matrix of an innovations model.

    For

    .. math::

       x_{t+1}=Ax_t+K\varepsilon_t,\qquad
       y_t=Cx_t+\varepsilon_t,\qquad
       \operatorname{cov}(\varepsilon_t)=V,

    the transfer function :math:`H(f)` gives

    .. math::

       S(f)=H(f)VH(f)^*/F_s.

    Parameters
    ----------
    system
        Batched or unbatched innovations-form process.
    frequencies
        One-dimensional frequency grid in cycles per unit time.
    sampling_frequency
        Sampling frequency ``Fs``. Frequencies must lie in ``[0, Fs/2]``.
    """
    if sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be positive")
    frequency = torch.as_tensor(
        frequencies,
        dtype=system.transition.dtype,
        device=system.transition.device,
    )
    if frequency.ndim != 1 or frequency.numel() == 0:
        raise ValueError("frequencies must be a non-empty one-dimensional tensor")
    if not bool(torch.isfinite(frequency).all().item()):
        raise ValueError("frequencies must contain only finite values")
    if bool(torch.any((frequency < 0) | (frequency > sampling_frequency / 2)).item()):
        raise ValueError("frequencies must lie in [0, sampling_frequency/2]")

    # innovations_transfer_function uses normalized cycles/sample internally.
    normalized = frequency / float(sampling_frequency)
    transfer = innovations_transfer_function(system, normalized)
    covariance = torch.as_tensor(
        system.innovation_covariance,
        dtype=system.transition.dtype,
        device=system.transition.device,
    )
    if transfer.ndim == 3:
        spectrum = (
            transfer
            @ covariance.to(transfer.dtype)
            @ transfer.conj().transpose(-1, -2)
        ) / float(sampling_frequency)
    else:
        if covariance.ndim == 2:
            covariance = covariance.unsqueeze(0)
        if covariance.shape[0] == 1 and transfer.shape[0] > 1:
            covariance = covariance.expand(transfer.shape[0], -1, -1)
        spectrum = (
            transfer
            @ covariance[:, None].to(transfer.dtype)
            @ transfer.conj().transpose(-1, -2)
        ) / float(sampling_frequency)
    return hermitian_part(spectrum)


@dataclass(frozen=True)
class SpectralMeasureContext:
    """Reusable full observation spectrum and its frequency metadata.

    The context is a lightweight cache for high-throughput model-derived
    measures that depend on the same full spectral-density matrix.  It does not
    alter any measure definition; it only avoids recomputing ``S(f)`` when
    several spectral features are evaluated on the same model and grid.
    """

    frequencies: torch.Tensor
    spectrum: torch.Tensor
    sampling_frequency: float

    @property
    def n_observations(self) -> int:
        """Number of observed variables represented by the spectrum."""
        return self.spectrum.shape[-1]

    def submatrix(self, indices) -> torch.Tensor:
        """Return one observation-group spectral submatrix."""
        index = torch.as_tensor(indices, dtype=torch.long, device=self.spectrum.device)
        return self.spectrum.index_select(-2, index).index_select(-1, index)

    def integrate(self, values: torch.Tensor, *, half_open: bool = False) -> torch.Tensor:
        """Integrate a spectral rate using this context's grid metadata."""
        return integrate_spectral_rate(
            values,
            self.frequencies,
            sampling_frequency=self.sampling_frequency,
            half_open=half_open,
        )


def build_spectral_measure_context(
    system: InnovationsStateSpace,
    frequencies: torch.Tensor,
    *,
    sampling_frequency: float = 1.0,
) -> SpectralMeasureContext:
    """Build a reusable full-spectrum context for model-derived measures."""
    frequency = torch.as_tensor(
        frequencies, dtype=system.transition.dtype, device=system.transition.device
    )
    spectrum = innovations_spectral_density(
        system, frequency, sampling_frequency=sampling_frequency
    )
    return SpectralMeasureContext(
        frequencies=frequency,
        spectrum=spectrum,
        sampling_frequency=float(sampling_frequency),
    )


def _resolve_spectral_measure_context(
    system: InnovationsStateSpace,
    frequencies: torch.Tensor,
    *,
    sampling_frequency: float = 1.0,
    context: SpectralMeasureContext | None = None,
) -> SpectralMeasureContext:
    """Return a compatible supplied context or build one."""
    if context is None:
        return build_spectral_measure_context(
            system, frequencies, sampling_frequency=sampling_frequency
        )
    frequency = torch.as_tensor(
        frequencies, dtype=context.frequencies.dtype, device=context.frequencies.device
    )
    if context.n_observations != system.observation.shape[-2]:
        raise ValueError("spectral context has incompatible observation dimension")
    if context.sampling_frequency != float(sampling_frequency):
        raise ValueError("spectral context has incompatible sampling_frequency")
    if frequency.shape != context.frequencies.shape or not torch.equal(
        frequency, context.frequencies
    ):
        raise ValueError("spectral context has incompatible frequencies")
    return context


def integrate_spectral_rate(
    values: torch.Tensor,
    frequencies: torch.Tensor,
    *,
    sampling_frequency: float = 1.0,
    half_open: bool = False,
) -> torch.Tensor:
    r"""Integrate a one-sided spectral information rate to a temporal rate.

    For real stationary processes,

    .. math::

       R = \frac{2}{F_s}\int_0^{F_s/2} r(f)\,df.

    ``half_open=True`` implements the exact discretisation used by the Faes
    OIR/HOP code: ``f_k = k Fs/(2N)``, ``k=0,...,N-1`` and the integral is the
    arithmetic mean across frequency bins.  Otherwise trapezoidal integration
    is used, which is appropriate for grids including both endpoints.
    """
    if sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be positive")
    value = torch.as_tensor(values)
    frequency = torch.as_tensor(
        frequencies, dtype=value.real.dtype, device=value.device
    )
    if frequency.ndim != 1 or frequency.numel() != value.shape[-1]:
        raise ValueError("frequencies must match the last dimension of values")
    if frequency.numel() < 2:
        raise ValueError("at least two frequency bins are required")
    if not bool(torch.all(frequency[1:] > frequency[:-1]).item()):
        raise ValueError("frequencies must be strictly increasing")

    if half_open:
        spacing = frequency[1:] - frequency[:-1]
        if not bool(
            torch.allclose(
                spacing,
                spacing[:1].expand_as(spacing),
                rtol=1e-6,
                atol=torch.finfo(frequency.dtype).eps * 16,
            )
        ):
            raise ValueError("half-open integration requires a uniform grid")
        expected_spacing = float(sampling_frequency) / (2.0 * frequency.numel())
        if not torch.isclose(
            spacing[0],
            torch.as_tensor(expected_spacing, dtype=frequency.dtype, device=frequency.device),
            rtol=1e-5,
            atol=torch.finfo(frequency.dtype).eps * 32,
        ):
            raise ValueError(
                "half-open grid must use spacing sampling_frequency/(2*N)"
            )
        if not torch.isclose(
            frequency[0],
            torch.zeros((), dtype=frequency.dtype, device=frequency.device),
            rtol=0.0,
            atol=torch.finfo(frequency.dtype).eps * 32,
        ):
            raise ValueError("half-open grid must start at zero")
        return value.mean(dim=-1)

    return (2.0 / float(sampling_frequency)) * torch.trapezoid(
        value, frequency, dim=-1
    )
