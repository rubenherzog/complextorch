"""Rosas--Mediano practical criteria and a ComplexTorch full-past extension.

The published finite-delay criteria from Rosas et al. (2020), Eq. (10), are
implemented for deterministic linear macro-features ``V_t = L X_t``. For
stationary Gaussian models they are evaluated exactly from model-implied
autocovariances, without simulation or refitting.

ComplexTorch additionally provides explicitly separate full-past analogues.
Those quantities are not definitions from Rosas et al.; they replace each
finite delayed source by its complete semi-infinite past and use exact
projected-history prediction covariances from the canonical innovations/DARE
core.

References
----------
- Rosas, F. E., Mediano, P. A. M., Jensen, H. J., Seth, A. K., Barrett, A. B.,
  Carhart-Harris, R. L., and Bor, D. (2020). Reconciling emergences: An
  information-theoretic approach to identify causal emergence in multivariate
  data. *PLoS Computational Biology*, 16(12), e1008289.
- ``pmediano/ReconcilingEmergences`` commit
  ``ecf591aacb6d58996c903b51a2f945cd7f713a32``.
"""
from __future__ import annotations

import math

import torch

from ..control import InnovationsStateSpace, _batched, _project_innovations_state_space
from ..linalg import solve_discrete_lyapunov, spd_logdet, symmetrise
from .backbone import CovarianceModel, as_innovations, observation_autocovariances
from .gaussian import gaussian_mutual_information


def _validate_base(base: float) -> float:
    """Validate and normalize the logarithm base."""
    value = float(base)
    if not math.isfinite(value) or value <= 0.0 or value == 1.0:
        raise ValueError("base must be finite, positive, and different from 1")
    return value


def _batched_autocovariances(
    autocovariances: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    """Normalize autocovariances to ``(batch, lag, n, n)``."""
    gamma = torch.as_tensor(autocovariances)
    single = gamma.ndim == 3
    if single:
        gamma = gamma.unsqueeze(0)
    if gamma.ndim != 4 or gamma.shape[-1] != gamma.shape[-2]:
        raise ValueError(
            "autocovariances must have shape (lag,n,n) or (batch,lag,n,n)"
        )
    return gamma, single


def _batched_projection(
    projection: torch.Tensor,
    covariance: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    """Validate and normalize a linear macro projection to batched form."""
    matrix = torch.as_tensor(
        projection, dtype=covariance.dtype, device=covariance.device
    )
    single = matrix.ndim == 2
    if single:
        matrix = matrix.unsqueeze(0)
    if matrix.ndim != 3:
        raise ValueError("macro_projection must have shape (m,n) or (batch,m,n)")
    if matrix.shape[-1] != covariance.shape[-1]:
        raise ValueError("macro_projection input dimension must match model dimension")
    if not 1 <= matrix.shape[-2] <= matrix.shape[-1]:
        raise ValueError("macro_projection output dimension must be between 1 and n")
    if not bool(torch.isfinite(matrix).all().item()):
        raise ValueError("macro_projection must contain only finite values")
    if bool(torch.any(torch.linalg.matrix_rank(matrix) < matrix.shape[-2]).item()):
        raise ValueError("macro_projection must have full row rank")
    return matrix, single


def _broadcast_model_projection(
    gamma: torch.Tensor,
    projection: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Broadcast model and projection over one compatible batch dimension."""
    batch = max(gamma.shape[0], projection.shape[0])
    if gamma.shape[0] not in (1, batch) or projection.shape[0] not in (1, batch):
        raise ValueError("incompatible model and macro_projection batch dimensions")
    if gamma.shape[0] == 1:
        gamma = gamma.expand(batch, *gamma.shape[1:])
    if projection.shape[0] == 1:
        projection = projection.expand(batch, *projection.shape[1:])
    return gamma, projection


def _scalar_vector_mi(
    scalar_variance: torch.Tensor,
    vector_covariance: torch.Tensor,
    vector_scalar_cross: torch.Tensor,
    *,
    base: float,
) -> torch.Tensor:
    """Vectorized ``I(scalar; vector)`` using the canonical Gaussian-MI core."""
    n = scalar_variance.shape[-1]
    m = vector_covariance.shape[-1]
    vector_covariance = vector_covariance.unsqueeze(-3).expand(-1, n, m, m)
    cross = vector_scalar_cross.transpose(-1, -2).unsqueeze(-2)
    top = torch.cat([scalar_variance[..., None, None], cross], dim=-1)
    bottom = torch.cat([cross.transpose(-1, -2), vector_covariance], dim=-1)
    joint = torch.cat([top, bottom], dim=-2)
    return gaussian_mutual_information(joint, 1, base=base)


def _pairwise_scalar_mi(
    variance: torch.Tensor,
    future_past_cross: torch.Tensor,
    *,
    base: float,
) -> torch.Tensor:
    """Return ``I(X_t^i; X_{t+tau}^j)`` for every ``(j, i)`` pair."""
    n = variance.shape[-1]
    past_variance = variance[:, None, :].expand(-1, n, -1)
    future_variance = variance[:, :, None].expand(-1, -1, n)
    joint = torch.stack(
        [
            torch.stack([past_variance, future_past_cross], dim=-1),
            torch.stack([future_past_cross, future_variance], dim=-1),
        ],
        dim=-2,
    )
    return gaussian_mutual_information(joint, 1, base=base)


def emergence_from_autocovariances(
    autocovariances: torch.Tensor,
    macro_projection: torch.Tensor,
    *,
    lag: int = 1,
    base: float = 2.0,
) -> dict[str, torch.Tensor]:
    r"""Evaluate the published Rosas--Mediano practical emergence criteria.

    For ``V_t = L X_t`` and :math:`\tau=\text{lag}` this computes

    .. math::

       \Psi_\tau(V)
       = I(V_t;V_{t+\tau})
       - \sum_j I(X_t^j;V_{t+\tau}),

    .. math::

       \Delta_\tau(V)
       = \max_j\left[
       I(V_t;X_{t+\tau}^j)
       - \sum_i I(X_t^i;X_{t+\tau}^j)\right],

    .. math::

       \Gamma_\tau(V)
       = \max_j I(V_t;X_{t+\tau}^j).

    ``autocovariances[..., tau, :, :]`` follows the ComplexTorch convention
    :math:`\operatorname{Cov}(X_{t+\tau}, X_t)`.
    """
    if lag < 1:
        raise ValueError("lag must be at least one")
    base = _validate_base(base)
    gamma, gamma_single = _batched_autocovariances(autocovariances)
    if gamma.shape[-3] <= lag:
        raise ValueError("autocovariances do not contain the requested lag")
    projection, projection_single = _batched_projection(
        macro_projection, gamma[:, 0]
    )
    gamma, projection = _broadcast_model_projection(gamma, projection)

    present = symmetrise(gamma[:, 0])
    future_past = gamma[:, lag]
    macro_covariance = symmetrise(
        projection @ present @ projection.transpose(-1, -2)
    )
    macro_future_past = projection @ future_past @ projection.transpose(-1, -2)
    macro_joint = torch.cat(
        [
            torch.cat(
                [macro_covariance, macro_future_past.transpose(-1, -2)], dim=-1
            ),
            torch.cat([macro_future_past, macro_covariance], dim=-1),
        ],
        dim=-2,
    )
    macro_mi = gaussian_mutual_information(
        macro_joint, macro_covariance.shape[-1], base=base
    )

    variances = torch.diagonal(present, dim1=-2, dim2=-1)

    macro_future_micro_past = projection @ future_past
    micro_to_macro = _scalar_vector_mi(
        variances,
        macro_covariance,
        macro_future_micro_past,
        base=base,
    )

    micro_future_macro_past = future_past @ projection.transpose(-1, -2)
    macro_to_micro = _scalar_vector_mi(
        variances,
        macro_covariance,
        micro_future_macro_past.transpose(-1, -2),
        base=base,
    )

    pairwise_micro = _pairwise_scalar_mi(variances, future_past, base=base)
    micro_sum_by_target = pairwise_micro.sum(dim=-1)

    psi = macro_mi - micro_to_macro.sum(dim=-1)
    delta_terms = macro_to_micro - micro_sum_by_target
    delta = delta_terms.max(dim=-1).values
    gamma_value = macro_to_micro.max(dim=-1).values

    result = {
        "psi": psi,
        "delta": delta,
        "gamma": gamma_value,
        "macro_mutual_information": macro_mi,
        "micro_to_macro_mutual_information": micro_to_macro,
        "macro_to_micro_mutual_information": macro_to_micro,
        "micro_pairwise_mutual_information": pairwise_micro,
    }
    if gamma_single and projection_single:
        return {name: value[0] for name, value in result.items()}
    return result


def _conditional_observation_covariance_from_past(
    model: CovarianceModel,
    conditioning_projection: torch.Tensor,
) -> torch.Tensor:
    """Return ``Cov(X_t | (L X)_{<t})`` via existing projected innovations.

    The canonical projection routine already solves the required generalized
    DARE. Its latent state is the steady-state predictor conditioned on the
    projected history. Therefore the conditional state-error covariance is
    the difference between the unconditional microscopic state covariance and
    the projected predictor-state covariance.
    """
    innovations = as_innovations(model)
    projected = _project_innovations_state_space(
        innovations, conditioning_projection
    )

    def _predictor_state_covariance(system: InnovationsStateSpace) -> torch.Tensor:
        """Return the stationary covariance of an innovations predictor state."""
        transition, _ = _batched(system.transition, 3)
        gain, _ = _batched(system.gain, 3)
        innovation_covariance, _ = _batched(system.innovation_covariance, 3)
        batch = max(
            transition.shape[0], gain.shape[0], innovation_covariance.shape[0]
        )
        values = [transition, gain, innovation_covariance]
        if any(value.shape[0] not in (1, batch) for value in values):
            raise ValueError("incompatible projected-innovations batch dimensions")
        transition, gain, innovation_covariance = [
            value.expand(batch, *value.shape[1:])
            if value.shape[0] == 1 else value
            for value in values
        ]
        process_covariance = symmetrise(
            gain @ innovation_covariance @ gain.transpose(-1, -2)
        )
        covariance, _ = solve_discrete_lyapunov(transition, process_covariance)
        return covariance

    microscopic_state = _predictor_state_covariance(innovations)
    projected_state = _predictor_state_covariance(projected)
    batch = max(microscopic_state.shape[0], projected_state.shape[0])
    if microscopic_state.shape[0] == 1:
        microscopic_state = microscopic_state.expand(batch, -1, -1)
    if projected_state.shape[0] == 1:
        projected_state = projected_state.expand(batch, -1, -1)

    observation, _ = _batched(innovations.observation, 3)
    noise, _ = _batched(innovations.innovation_covariance, 3)
    if observation.shape[0] == 1:
        observation = observation.expand(batch, -1, -1)
    if noise.shape[0] == 1:
        noise = noise.expand(batch, -1, -1)
    state_error = symmetrise(microscopic_state - projected_state)
    return symmetrise(
        observation @ state_error @ observation.transpose(-1, -2) + noise
    )


def _expanded_innovations_for_singleton_histories(
    model: CovarianceModel,
    batch: int,
    n_variables: int,
) -> InnovationsStateSpace:
    """Repeat each model once per singleton conditioning history."""
    innovations = as_innovations(model)
    values = []
    for tensor in (
        innovations.transition,
        innovations.observation,
        innovations.gain,
        innovations.innovation_covariance,
    ):
        value, _ = _batched(tensor, 3)
        if value.shape[0] not in (1, batch):
            raise ValueError("model batch is incompatible with macro_projection batch")
        if value.shape[0] == 1:
            value = value.expand(batch, *value.shape[1:])
        values.append(value.repeat_interleave(n_variables, dim=0))
    return InnovationsStateSpace(*values)


def emergence_from_full_past(
    model: CovarianceModel,
    macro_projection: torch.Tensor,
    *,
    base: float = 2.0,
    observation_covariance: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    r"""Evaluate the ComplexTorch full-past extension of emergence criteria.

    This extension is **not** defined in Rosas et al. (2020). ComplexTorch
    defines

    .. math::

       \Psi_\infty(V)
       = I(V_{<t};V_t)-\sum_j I(X^j_{<t};V_t),

    .. math::

       \Delta_\infty(V)
       = \max_j\left[I(V_{<t};X_t^j)
       -\sum_i I(X^i_{<t};X_t^j)\right],

    .. math::

       \Gamma_\infty(V)=\max_j I(V_{<t};X_t^j).

    Every conditional covariance is computed exactly from the model's
    innovations representation using the existing generalized-DARE projection
    core. No finite-history truncation or simulated trajectory is used.
    """
    base = _validate_base(base)
    covariance = (
        observation_autocovariances(model, 0)[..., 0, :, :]
        if observation_covariance is None
        else torch.as_tensor(observation_covariance)
    )
    covariance = torch.as_tensor(covariance)
    covariance_single = covariance.ndim == 2
    if covariance_single:
        covariance = covariance.unsqueeze(0)
    projection, projection_single = _batched_projection(
        macro_projection, covariance
    )
    gamma = covariance.unsqueeze(1)
    gamma, projection = _broadcast_model_projection(gamma, projection)
    covariance = symmetrise(gamma[:, 0])
    batch, n_variables, _ = covariance.shape

    macro_covariance = symmetrise(
        projection @ covariance @ projection.transpose(-1, -2)
    )

    macro_conditioned = _conditional_observation_covariance_from_past(
        model, projection
    )
    if macro_conditioned.ndim == 2:
        macro_conditioned = macro_conditioned.unsqueeze(0)
    if macro_conditioned.shape[0] == 1 and batch > 1:
        macro_conditioned = macro_conditioned.expand(batch, -1, -1)
    macro_given_macro_past = symmetrise(
        projection @ macro_conditioned @ projection.transpose(-1, -2)
    )
    macro_mi = 0.5 * (
        spd_logdet(macro_covariance) - spd_logdet(macro_given_macro_past)
    ) / math.log(base)

    marginal_variance = torch.diagonal(covariance, dim1=-2, dim2=-1)
    macro_conditioned_variance = torch.diagonal(
        macro_conditioned, dim1=-2, dim2=-1
    )
    macro_to_micro = 0.5 * torch.log(
        marginal_variance / macro_conditioned_variance
    ) / math.log(base)

    selectors = torch.eye(
        n_variables, dtype=covariance.dtype, device=covariance.device
    ).reshape(1, n_variables, 1, n_variables).expand(batch, -1, -1, -1)
    selectors = selectors.reshape(batch * n_variables, 1, n_variables)
    expanded_model = _expanded_innovations_for_singleton_histories(
        model, batch, n_variables
    )
    micro_conditioned = _conditional_observation_covariance_from_past(
        expanded_model, selectors
    ).reshape(batch, n_variables, n_variables, n_variables)

    conditional_variance = torch.diagonal(
        micro_conditioned, dim1=-2, dim2=-1
    )
    micro_pairwise = 0.5 * torch.log(
        marginal_variance[:, None, :] / conditional_variance
    ) / math.log(base)
    micro_pairwise = micro_pairwise.transpose(-1, -2)

    macro_conditioned_by_micro = symmetrise(
        projection[:, None]
        @ micro_conditioned
        @ projection[:, None].transpose(-1, -2)
    )
    micro_to_macro = 0.5 * (
        spd_logdet(macro_covariance)[:, None]
        - spd_logdet(macro_conditioned_by_micro)
    ) / math.log(base)

    psi = macro_mi - micro_to_macro.sum(dim=-1)
    delta_terms = macro_to_micro - micro_pairwise.sum(dim=-1)
    delta = delta_terms.max(dim=-1).values
    gamma_value = macro_to_micro.max(dim=-1).values

    result = {
        "psi": psi,
        "delta": delta,
        "gamma": gamma_value,
        "macro_mutual_information": macro_mi,
        "micro_to_macro_mutual_information": micro_to_macro,
        "macro_to_micro_mutual_information": macro_to_micro,
        "micro_pairwise_mutual_information": micro_pairwise,
    }
    if covariance_single and projection_single:
        return {name: value[0] for name, value in result.items()}
    return result


def emergence_from_model(
    model: CovarianceModel,
    macro_projection: torch.Tensor,
    *,
    lag: int = 1,
    history: str = "lagged",
    base: float = 2.0,
    autocovariance_sequence: torch.Tensor | None = None,
    observation_covariance: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    r"""Return finite-delay or full-past emergence criteria.

    ``history="lagged"`` implements the practical :math:`\Psi_\tau`,
    :math:`\Delta_\tau`, and :math:`\Gamma_\tau` criteria published by
    Rosas--Mediano et al. ``history="full"`` selects the explicitly
    ComplexTorch-defined full-past extension :math:`\Psi_\infty`,
    :math:`\Delta_\infty`, and :math:`\Gamma_\infty`.

    Parameters
    ----------
    model
        Stationary ``VARSystem``, ``StateSpaceModel``, or
        ``InnovationsStateSpace`` supported by the covariance backbone.
    macro_projection
        Full-row-rank linear map ``L`` in ``V_t = L X_t``, with shape
        ``(m,n)`` or ``(batch,m,n)``. No projection is estimated internally.
    lag
        Positive delay :math:`\tau` for ``history="lagged"``. It has no role
        in the full-past definition.
    history
        ``"lagged"`` for the published Rosas--Mediano criteria or ``"full"``
        for the ComplexTorch extension.
    base
        Logarithm base. Defaults to 2 (bits).
    autocovariance_sequence
        Optional precomputed autocovariances for the finite-delay path.
    observation_covariance
        Optional precomputed stationary covariance for the full-past path.
    """
    if history not in {"lagged", "full"}:
        raise ValueError("history must be 'lagged' or 'full'")
    if lag < 1:
        raise ValueError("lag must be at least one")
    if history == "full":
        return emergence_from_full_past(
            model,
            macro_projection,
            base=base,
            observation_covariance=observation_covariance,
        )
    autocovariances = (
        observation_autocovariances(model, lag)
        if autocovariance_sequence is None
        else autocovariance_sequence
    )
    return emergence_from_autocovariances(
        autocovariances, macro_projection, lag=lag, base=base
    )


def emergence_measures(
    model: CovarianceModel,
    macro_projection: torch.Tensor,
    *,
    lag: int = 1,
    history: str = "lagged",
    base: float = 2.0,
) -> dict[str, torch.Tensor]:
    """Backward-compatible alias for :func:`emergence_from_model`."""
    return emergence_from_model(
        model, macro_projection, lag=lag, history=history, base=base
    )


def emergence_from_observations(
    observations: torch.Tensor,
    macro_projection: torch.Tensor,
    *,
    lag: int = 1,
    base: float = 2.0,
) -> dict[str, torch.Tensor]:
    """Gaussian plug-in estimate of the published finite-delay criteria.

    This secondary estimator intentionally implements only the finite-delay
    Rosas--Mediano quantities. The exact ``history="full"`` extension is
    model-based because it requires an innovations/DARE representation.
    """
    if lag < 1:
        raise ValueError("lag must be at least one")
    x = torch.as_tensor(observations)
    if x.ndim != 2:
        raise ValueError("observations must have shape (time, variables)")
    if x.shape[0] <= lag + 1:
        raise ValueError("observations are too short for the requested lag")
    if not x.is_floating_point():
        raise TypeError("observations must use a floating-point dtype")
    if not bool(torch.isfinite(x).all().item()):
        raise ValueError("observations must contain only finite values")

    centered = x - x.mean(dim=0)
    denominator = x.shape[0] - 1
    gamma0 = centered.transpose(-1, -2) @ centered / denominator
    future = centered[lag:]
    past = centered[:-lag]
    gamma_lag = future.transpose(-1, -2) @ past / (future.shape[0] - 1)
    autocovariances = torch.stack([gamma0, gamma_lag], dim=0)
    if lag > 1:
        padded = torch.zeros(
            (lag + 1, x.shape[-1], x.shape[-1]), dtype=x.dtype, device=x.device
        )
        padded[0] = gamma0
        padded[lag] = gamma_lag
        autocovariances = padded
    return emergence_from_autocovariances(
        autocovariances, macro_projection, lag=lag, base=base
    )


__all__ = [
    "emergence_from_autocovariances",
    "emergence_from_full_past",
    "emergence_from_model",
    "emergence_measures",
    "emergence_from_observations",
]
