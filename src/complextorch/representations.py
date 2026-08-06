r"""Canonical VAR and linear Gaussian state-space representations.

A VAR(p) process is

.. math::

   x_t = \sum_{k=1}^{p} A_k x_{t-k} + \varepsilon_t,
   \qquad \varepsilon_t \sim \mathcal N(0,\Sigma).

The companion representation embeds this recursion into a first-order state
transition and provides the bridge to state-space calculations.

References
----------
- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*.
- Barnett, L. and Seth, A. K. (2015). Granger causality for state-space models.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .linalg import LyapunovInfo, solve_discrete_lyapunov, spectral_radius


@dataclass(frozen=True)
class StateSpaceModel:
    """Linear Gaussian state-space model.

    The optional ``state_covariance`` stores the stationary latent covariance
    used by analytical observation-covariance and autocovariance calculations.
    """

    transition: torch.Tensor
    observation: torch.Tensor
    process_covariance: torch.Tensor
    observation_covariance: torch.Tensor
    state_covariance: torch.Tensor | None = None
    sampling_frequency: float | None = None
    channel_names: tuple[str, ...] | None = None

    @property
    def spectral_radius(self) -> torch.Tensor:
        """Return the largest absolute eigenvalue of the transition matrix."""

        return spectral_radius(self.transition)


@dataclass(frozen=True)
class VARSystem:
    """Stationary Gaussian VAR process and its companion-state primitives."""

    coefficients: torch.Tensor
    innovation_covariance: torch.Tensor
    companion: torch.Tensor
    companion_noise_covariance: torch.Tensor
    state_covariance: torch.Tensor
    projection: torch.Tensor
    present_covariance: torch.Tensor
    spectral_radius: torch.Tensor
    lyapunov_info: LyapunovInfo

    @property
    def batch_size(self) -> int:
        """Return the number of independently represented VAR systems."""

        return int(self.coefficients.shape[0])

    @property
    def order(self) -> int:
        """Return the autoregressive order ``p``."""

        return int(self.coefficients.shape[1])

    @property
    def n_variables(self) -> int:
        """Return the number of observed variables."""

        return int(self.coefficients.shape[2])

    def to_state_space(self) -> StateSpaceModel:
        """Return the exactly equivalent companion state-space model.

        The observation matrix selects the present VAR block from the companion
        state, and the observation-noise covariance is zero because the VAR
        innovations enter through the companion process noise.
        """

        batch = self.companion.shape[0]
        n_variables = self.n_variables
        zero = torch.zeros(
            (batch, n_variables, n_variables),
            dtype=self.companion.dtype,
            device=self.companion.device,
        )
        return StateSpaceModel(
            self.companion,
            self.projection,
            self.companion_noise_covariance,
            zero,
            self.state_covariance,
        )


def _normalise_coefficients(
    coefficients: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    """Normalize VAR coefficients to ``(batch, lag, target, source)``.

    Returns the normalized tensor and a flag indicating whether the original
    input was unbatched.
    """

    coefficient_tensor = torch.as_tensor(coefficients)
    unbatched = coefficient_tensor.ndim == 3
    if unbatched:
        coefficient_tensor = coefficient_tensor.unsqueeze(0)
    if (
        coefficient_tensor.ndim != 4
        or coefficient_tensor.shape[-1] != coefficient_tensor.shape[-2]
    ):
        raise ValueError(
            "coefficients must have shape (p,n,n) or (batch,p,n,n)"
        )
    return coefficient_tensor, unbatched


def companion_matrix(coefficients: torch.Tensor) -> torch.Tensor:
    r"""Construct the first-order companion transition of a VAR(p).

    The first block row contains ``[A_1, ..., A_p]`` and lower block rows shift
    lagged observations forward by one step. See Lütkepohl (2005).
    """
    # Stack VAR lags into a first-order Markov state whose top block contains the autoregressive coefficients.

    coefficient_tensor, unbatched = _normalise_coefficients(coefficients)
    batch, order, n_variables, _ = coefficient_tensor.shape
    companion = torch.zeros(
        (batch, order * n_variables, order * n_variables),
        dtype=coefficient_tensor.dtype,
        device=coefficient_tensor.device,
    )
    companion[:, :n_variables, :] = coefficient_tensor.permute(
        0, 2, 1, 3
    ).reshape(batch, n_variables, order * n_variables)
    if order > 1:
        companion[:, n_variables:, : (order - 1) * n_variables] = torch.eye(
            (order - 1) * n_variables,
            dtype=coefficient_tensor.dtype,
            device=coefficient_tensor.device,
        )
    return companion[0] if unbatched else companion


def build_var_system(
    coefficients: torch.Tensor,
    innovation_covariance: torch.Tensor,
    *,
    lyapunov_method: str = "doubling",
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> VARSystem:
    r"""Build all stationary companion primitives for a Gaussian VAR.

    The companion state covariance :math:`P` solves

    .. math::

       P = A_c P A_c^\top + Q_c.

    Parameters
    ----------
    coefficients
        VAR coefficients in batched or unbatched ComplexTorch layout.
    innovation_covariance
        Innovation covariance :math:`\Sigma`.
    lyapunov_method
        Numerical method passed to the discrete Lyapunov solver.
    rtol, atol
        Relative and absolute convergence tolerances.

    Returns
    -------
    VARSystem
        Canonical stationary representation used by analytical measures.

    References
    ----------
    - Lütkepohl (2005).
    - Barnett and Seth (2015).
    """
    # Derive companion-state covariance and stationary observation quantities from fitted VAR parameters.

    coefficient_tensor, _ = _normalise_coefficients(coefficients)
    covariance = torch.as_tensor(
        innovation_covariance,
        dtype=coefficient_tensor.dtype,
        device=coefficient_tensor.device,
    )
    if covariance.ndim == 2:
        covariance = covariance.unsqueeze(0)
    if covariance.shape[0] == 1 and coefficient_tensor.shape[0] > 1:
        covariance = covariance.expand(
            coefficient_tensor.shape[0], -1, -1
        ).contiguous()
    if (
        covariance.ndim != 3
        or covariance.shape[1:] != coefficient_tensor.shape[2:]
        or covariance.shape[0] != coefficient_tensor.shape[0]
    ):
        raise ValueError("invalid innovation covariance shape")

    companion = companion_matrix(coefficient_tensor)
    batch, state_dimension, _ = companion.shape
    n_variables = coefficient_tensor.shape[-1]
    companion_noise = torch.zeros_like(companion)
    companion_noise[:, :n_variables, :n_variables] = covariance

    # The stationary companion covariance is the unique stable Lyapunov solution.
    state_covariance, lyapunov_info = solve_discrete_lyapunov(
        companion,
        companion_noise,
        method=lyapunov_method,
        rtol=rtol,
        atol=atol,
    )
    projection = torch.zeros(
        (batch, n_variables, state_dimension),
        dtype=coefficient_tensor.dtype,
        device=coefficient_tensor.device,
    )
    projection[:, :, :n_variables] = torch.eye(
        n_variables,
        dtype=coefficient_tensor.dtype,
        device=coefficient_tensor.device,
    )
    present_covariance = (
        projection @ state_covariance @ projection.transpose(-1, -2)
    )
    return VARSystem(
        coefficient_tensor,
        covariance,
        companion,
        companion_noise,
        state_covariance,
        projection,
        present_covariance,
        spectral_radius(companion),
        lyapunov_info,
    )
