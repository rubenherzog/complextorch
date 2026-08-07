r"""Inference for linear Gaussian state-space systems.

The module implements Kalman filtering, Rauch--Tung--Striebel smoothing,
N4SID subspace identification, and expectation--maximisation refinement for
models of the form

.. math::

   z_{t+1}=A z_t+w_t,\qquad w_t\sim\mathcal N(0,Q),

.. math::

   y_t=C z_t+v_t,\qquad v_t\sim\mathcal N(0,R).

References
----------
- Kalman, R. E. (1960). A new approach to linear filtering and prediction.
- Rauch, H. E., Tung, F., and Striebel, C. T. (1965). Maximum-likelihood
  estimates of linear dynamic systems.
- Van Overschee, P. and De Moor, B. (1994). N4SID.
- Shumway, R. H. and Stoffer, D. S. (1982). EM for time-series smoothing.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from sklearn.base import BaseEstimator

from .linalg import symmetrise
from .representations import StateSpaceModel
from .control import InnovationsStateSpace
from ._subspace import _block_hankel, _larimore_decomposition, _resolve_dtype


@dataclass(frozen=True)
class KalmanResult:
    """Filtered and predicted moments from a Kalman recursion.

    Attributes
    ----------
    filtered_mean, filtered_covariance
        Posterior state moments :math:`p(z_t\mid y_{1:t})`.
    predicted_mean, predicted_covariance
        One-step predictive moments :math:`p(z_t\mid y_{1:t-1})`.
    innovations, innovation_covariance
        Observation prediction errors and their covariance matrices.
    log_likelihood
        Exact Gaussian log likelihood accumulated over time.
    """

    filtered_mean: torch.Tensor
    filtered_covariance: torch.Tensor
    predicted_mean: torch.Tensor
    predicted_covariance: torch.Tensor
    innovations: torch.Tensor
    innovation_covariance: torch.Tensor
    log_likelihood: torch.Tensor


def _as_2d(observations: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Normalize observations to ``(batch, time, variables)``.

    Returns the normalized tensor and a flag indicating whether the original
    input was unbatched.
    """

    tensor = torch.as_tensor(observations)
    single = tensor.ndim == 2
    if single:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(
            "observations must have shape (time,variables) or "
            "(batch,time,variables)"
        )
    return tensor, single


def kalman_filter(
    observations,
    system,
    *,
    initial_mean=None,
    initial_covariance=None,
):
    r"""Run a batched Kalman filter with exact Gaussian log likelihood.

    For prediction moments :math:`m_t^-` and :math:`P_t^-`,

    .. math::

       m_t^- = A m_{t-1},\qquad
       P_t^- = A P_{t-1}A^\top+Q.

    With innovation :math:`e_t=y_t-Cm_t^-` and
    :math:`S_t=CP_t^-C^\top+R`, the gain is

    .. math::

       K_t=P_t^-C^\top S_t^{-1}.

    The posterior covariance uses the Joseph stabilized form rather than the
    algebraically shorter subtraction form.

    Parameters
    ----------
    observations
        Observations in ``(time, variables)`` or batch-first layout.
    system
        Linear Gaussian state-space model.
    initial_mean, initial_covariance
        Optional initial latent moments. When omitted, zero mean and the
        supplied stationary covariance, or identity covariance, are used.

    Returns
    -------
    KalmanResult
        Filtered moments, innovations and exact log likelihood.

    References
    ----------
    - Kalman (1960).
    - Anderson, B. D. O. and Moore, J. B. (1979). *Optimal Filtering*.
    """
    # Alternate state prediction and measurement update using the innovation covariance and Kalman gain.

    observations_tensor, single = _as_2d(observations)
    transition = (
        system.transition.unsqueeze(0)
        if system.transition.ndim == 2
        else system.transition
    )
    observation = (
        system.observation.unsqueeze(0)
        if system.observation.ndim == 2
        else system.observation
    )
    process_covariance = (
        system.process_covariance.unsqueeze(0)
        if system.process_covariance.ndim == 2
        else system.process_covariance
    )
    observation_covariance = (
        system.observation_covariance.unsqueeze(0)
        if system.observation_covariance.ndim == 2
        else system.observation_covariance
    )

    batch = observations_tensor.shape[0]
    transition, observation, process_covariance, observation_covariance = [
        value.expand(batch, *value.shape[1:]) if value.shape[0] == 1 else value
        for value in (
            transition,
            observation,
            process_covariance,
            observation_covariance,
        )
    ]
    state_dimension = transition.shape[-1]
    observation_dimension = observation.shape[-2]

    if initial_mean is None:
        mean = torch.zeros(
            (batch, state_dimension),
            dtype=observations_tensor.dtype,
            device=observations_tensor.device,
        )
    else:
        mean = torch.as_tensor(
            initial_mean,
            dtype=observations_tensor.dtype,
            device=observations_tensor.device,
        ).expand(batch, -1)

    if initial_covariance is None:
        covariance = system.state_covariance
        if covariance is None:
            covariance = torch.eye(
                state_dimension,
                dtype=observations_tensor.dtype,
                device=observations_tensor.device,
            ).expand(batch, state_dimension, state_dimension)
        elif covariance.ndim == 2:
            covariance = covariance.unsqueeze(0).expand(
                batch, state_dimension, state_dimension
            )
    else:
        covariance = torch.as_tensor(
            initial_covariance,
            dtype=observations_tensor.dtype,
            device=observations_tensor.device,
        )
        if covariance.ndim == 2:
            covariance = covariance.unsqueeze(0)

    filtered_means = []
    filtered_covariances = []
    predicted_means = []
    predicted_covariances = []
    innovations = []
    innovation_covariances = []
    log_likelihood = torch.zeros(
        batch,
        dtype=observations_tensor.dtype,
        device=observations_tensor.device,
    )
    log_two_pi = torch.log(
        torch.tensor(
            2.0 * torch.pi,
            dtype=observations_tensor.dtype,
            device=observations_tensor.device,
        )
    )

    for time_index in range(observations_tensor.shape[1]):
        predicted_mean = torch.einsum("bij,bj->bi", transition, mean)
        predicted_covariance = symmetrise(
            transition @ covariance @ transition.transpose(-1, -2)
            + process_covariance
        )
        innovation = observations_tensor[:, time_index] - torch.einsum(
            "bmd,bd->bm", observation, predicted_mean
        )
        innovation_covariance = symmetrise(
            observation
            @ predicted_covariance
            @ observation.transpose(-1, -2)
            + observation_covariance
        )

        # Cholesky solves avoid explicitly inverting the innovation covariance.
        # Factor the positive-definite covariance so whitening and solves use stable triangular algebra.
        cholesky = torch.linalg.cholesky(innovation_covariance)
        gain = torch.cholesky_solve(
            (
                predicted_covariance
                @ observation.transpose(-1, -2)
            ).transpose(-1, -2),
            cholesky,
        ).transpose(-1, -2)
        mean = predicted_mean + torch.einsum(
            "bdm,bm->bd", gain, innovation
        )

        # Joseph form maintains symmetry and positive semi-definiteness under
        # finite precision: (I-KC)P-(I-KC)' + KRK'.
        identity = torch.eye(
            state_dimension,
            dtype=observations_tensor.dtype,
            device=observations_tensor.device,
        ).expand(batch, state_dimension, state_dimension)
        correction = identity - gain @ observation
        covariance = symmetrise(
            correction
            @ predicted_covariance
            @ correction.transpose(-1, -2)
            + gain
            @ observation_covariance
            @ gain.transpose(-1, -2)
        )

        quadratic = torch.cholesky_solve(
            innovation.unsqueeze(-1), cholesky
        ).squeeze(-1)
        log_likelihood += -0.5 * (
            observation_dimension * log_two_pi
            + 2.0
            * torch.log(
                torch.diagonal(cholesky, dim1=-2, dim2=-1)
            ).sum(-1)
            + (innovation * quadratic).sum(-1)
        )

        predicted_means.append(predicted_mean)
        predicted_covariances.append(predicted_covariance)
        filtered_means.append(mean)
        filtered_covariances.append(covariance)
        innovations.append(innovation)
        innovation_covariances.append(innovation_covariance)

    stack = lambda values: torch.stack(values, dim=1)
    result = KalmanResult(
        stack(filtered_means),
        stack(filtered_covariances),
        stack(predicted_means),
        stack(predicted_covariances),
        stack(innovations),
        stack(innovation_covariances),
        log_likelihood,
    )
    if single:
        result = KalmanResult(
            *(getattr(result, field)[0] for field in result.__dataclass_fields__)
        )
    return result


@dataclass(frozen=True)
class SmootherResult:
    """Smoothed latent moments from the RTS backward recursion."""

    smoothed_mean: torch.Tensor
    smoothed_covariance: torch.Tensor
    lag_covariance: torch.Tensor
    log_likelihood: torch.Tensor


def kalman_smoother(observations, system):
    r"""Run Rauch--Tung--Striebel smoothing after Kalman filtering.

    The backward smoothing gain is

    .. math::

       J_t=P_{t\mid t}A^\top(P_{t+1\mid t})^{-1}.

    Parameters
    ----------
    observations
        Observations in unbatched or batch-first layout.
    system
        Linear Gaussian state-space model.

    Returns
    -------
    SmootherResult
        Smoothed means, covariances and adjacent-time cross covariances.

    References
    ----------
    - Rauch, Tung, and Striebel (1965).
    """
    # Run the Rauch-Tung-Striebel backward recursion to condition filtered states on all observations.

    filtered = kalman_filter(observations, system)
    single = filtered.filtered_mean.ndim == 2
    filtered_mean = (
        filtered.filtered_mean.unsqueeze(0)
        if single
        else filtered.filtered_mean
    )
    filtered_covariance = (
        filtered.filtered_covariance.unsqueeze(0)
        if single
        else filtered.filtered_covariance
    )
    predicted_mean = (
        filtered.predicted_mean.unsqueeze(0)
        if single
        else filtered.predicted_mean
    )
    predicted_covariance = (
        filtered.predicted_covariance.unsqueeze(0)
        if single
        else filtered.predicted_covariance
    )
    transition = (
        system.transition.unsqueeze(0)
        if system.transition.ndim == 2
        else system.transition
    ).expand(filtered_mean.shape[0], -1, -1)

    smoothed_mean = filtered_mean.clone()
    smoothed_covariance = filtered_covariance.clone()
    lag_covariance = torch.zeros_like(smoothed_covariance)

    for time_index in range(filtered_mean.shape[1] - 2, -1, -1):
        # Solve P_pred * X = A * P_filt and transpose to obtain the RTS gain,
        # avoiding an explicit inverse of the predicted covariance.
        # Solve the linear system directly instead of multiplying by an explicit inverse.
        smoother_gain = torch.linalg.solve(
            predicted_covariance[:, time_index + 1],
            transition @ filtered_covariance[:, time_index],
        ).transpose(-1, -2)
        smoothed_mean[:, time_index] = filtered_mean[:, time_index] + torch.einsum(
            "bij,bj->bi",
            smoother_gain,
            smoothed_mean[:, time_index + 1]
            - predicted_mean[:, time_index + 1],
        )
        smoothed_covariance[:, time_index] = symmetrise(
            filtered_covariance[:, time_index]
            + smoother_gain
            @ (
                smoothed_covariance[:, time_index + 1]
                - predicted_covariance[:, time_index + 1]
            )
            @ smoother_gain.transpose(-1, -2)
        )
        lag_covariance[:, time_index + 1] = (
            smoothed_covariance[:, time_index + 1]
            @ smoother_gain.transpose(-1, -2)
        )

    result = SmootherResult(
        smoothed_mean,
        smoothed_covariance,
        lag_covariance,
        filtered.log_likelihood.unsqueeze(0)
        if single
        else filtered.log_likelihood,
    )
    if single:
        return SmootherResult(
            result.smoothed_mean[0],
            result.smoothed_covariance[0],
            result.lag_covariance[0],
            result.log_likelihood[0],
        )
    return result


class N4SID(BaseEstimator):
    r"""Estimate linear Gaussian state-space systems by compact N4SID.

    The estimator accepts one trajectory ``(time, variables)`` or batch-first
    observations ``(batch, time, variables)``. In ``mode="pooled"`` all
    trajectories identify one common system, while preserving trial boundaries
    when estimating state transitions. In ``mode="independent"`` one system is
    estimated per trajectory using batched Torch linear algebra.

    References
    ----------
    - Van Overschee, P. and De Moor, B. (1994). N4SID.
    """

    def __init__(
        self,
        n_states: int,
        block_rows: int = 10,
        ridge: float = 1e-8,
        *,
        mode: str = "pooled",
        device: str | torch.device = "auto",
        dtype: str | torch.dtype = "float64",
    ):
        """Initialize N4SID settings."""
        self.n_states = n_states
        self.block_rows = block_rows
        self.ridge = ridge
        self.mode = mode
        self.device = device
        self.dtype = dtype

    def fit(self, observations, y=None):
        """Fit one pooled system or one system per trajectory.

        Parameters
        ----------
        observations
            Array with shape ``(time, variables)`` or
            ``(batch, time, variables)``.
        y
            Unused scikit-learn compatibility target.
        """
        del y
        values, single = _normalise_ss_observations(
            observations, device=self.device, dtype=self.dtype
        )
        if self.n_states < 1 or self.block_rows < 1 or self.ridge < 0:
            raise ValueError("invalid N4SID settings")
        if self.mode not in {"pooled", "independent"}:
            raise ValueError("mode must be 'pooled' or 'independent'")
        batch, n_times, n_variables = values.shape
        if n_times <= 2 * self.block_rows + 2:
            raise ValueError("not enough samples for requested block_rows")
        values = values - values.mean(dim=1, keepdim=True)
        past, future = _block_hankel(values, self.block_rows, self.block_rows)
        n_columns = past.shape[-1]

        if self.mode == "pooled":
            past_fit = past.permute(1, 0, 2).reshape(past.shape[1], -1).unsqueeze(0)
            future_fit = future.permute(1, 0, 2).reshape(future.shape[1], -1).unsqueeze(0)
        else:
            past_fit, future_fit = past, future

        identity = torch.eye(
            past_fit.shape[-2], dtype=values.dtype, device=values.device
        )
        gram = past_fit @ past_fit.transpose(-1, -2) + self.ridge * identity
        projection = (
            future_fit
            @ past_fit.transpose(-1, -2)
            @ torch.linalg.pinv(gram)
            @ past_fit
        )
        # Extract orthogonal latent modes; singular values quantify the variance or canonical dependence retained by each mode.
        left, singular_values, _ = torch.linalg.svd(projection, full_matrices=False)
        if self.n_states > singular_values.shape[-1]:
            raise ValueError("n_states exceeds identifiable subspace rank")
        observability = left[..., : self.n_states] * torch.sqrt(
            singular_values[..., : self.n_states]
        ).unsqueeze(-2)

        if self.mode == "pooled":
            common_observability = observability.expand(batch, -1, -1)
            state_columns = torch.linalg.lstsq(
                common_observability, future
            ).solution
        else:
            state_columns = torch.linalg.lstsq(observability, future).solution
        states = state_columns.transpose(-1, -2)
        transition, observation, process_covariance, observation_covariance = (
            _fit_general_state_space_from_states(
                values,
                states,
                observation_start=self.block_rows,
                mode=self.mode,
                min_covar=max(self.ridge, 1e-12),
            )
        )
        state_covariance = (
            _pooled_covariance(states) if self.mode == "pooled" else _batch_covariance(states)
        )
        if single and self.mode == "independent":
            transition = transition[0]
            observation = observation[0]
            process_covariance = process_covariance[0]
            observation_covariance = observation_covariance[0]
            state_covariance = state_covariance[0]
        self.system_ = StateSpaceModel(
            transition,
            observation,
            process_covariance,
            observation_covariance,
            state_covariance=state_covariance,
        )
        self.transition_ = transition
        self.observation_ = observation
        self.process_covariance_ = process_covariance
        self.observation_covariance_ = observation_covariance
        self.singular_values_ = singular_values.squeeze(0) if self.mode == "pooled" else singular_values
        self.states_ = states[0] if single else states
        self.n_states_ = self.n_states
        return self


class LarimoreStateSpace(BaseEstimator):
    r"""Estimate an innovations-form state-space model by Larimore CVA.

    The fitted model is

    .. math::

       z_{t+1}=Az_t+K\varepsilon_t,\qquad
       y_t=Cz_t+\varepsilon_t,\quad
       \varepsilon_t\sim\mathcal N(0,V).

    Batch semantics match :class:`N4SID`: pooled mode estimates one common
    system without connecting trial boundaries, while independent mode returns
    a batched collection of systems.

    References
    ----------
    - Larimore, W. E. (1990, 1996).
    - Bauer, D. (2001), for the associated order-selection criterion.
    - ComplexBox ``mvgc.ss.tsdata_to_ss``.
    """

    def __init__(
        self,
        n_states: int,
        past_horizon: int,
        *,
        future_horizon: int | None = None,
        ridge: float = 1e-12,
        covariance: str = "mle",
        mode: str = "pooled",
        device: str | torch.device = "auto",
        dtype: str | torch.dtype = "float64",
    ):
        """Initialize Larimore state-space identification settings."""
        self.n_states = n_states
        self.past_horizon = past_horizon
        self.future_horizon = future_horizon
        self.ridge = ridge
        self.covariance = covariance
        self.mode = mode
        self.device = device
        self.dtype = dtype

    def fit(self, observations, y=None):
        """Estimate ``A``, ``C``, ``K`` and ``V`` from observations."""
        del y
        values, single = _normalise_ss_observations(
            observations, device=self.device, dtype=self.dtype
        )
        future_horizon = (
            self.past_horizon
            if self.future_horizon is None
            else int(self.future_horizon)
        )
        if self.n_states < 1 or self.past_horizon < 1 or future_horizon < 1:
            raise ValueError("state dimension and horizons must be positive")
        if self.ridge < 0:
            raise ValueError("ridge must be non-negative")
        if self.mode not in {"pooled", "independent"}:
            raise ValueError("mode must be 'pooled' or 'independent'")
        if self.covariance not in {"mle", "unbiased"}:
            raise ValueError("covariance must be 'mle' or 'unbiased'")
        if self.past_horizon + future_horizon > values.shape[1]:
            raise ValueError("past/future horizons are too large for the series")

        values = values - values.mean(dim=1, keepdim=True)
        past, future = _block_hankel(values, self.past_horizon, future_horizon)
        batch, _, n_columns = past.shape
        if self.mode == "pooled":
            past_fit = past.permute(1, 0, 2).reshape(past.shape[1], -1).unsqueeze(0)
            future_fit = future.permute(1, 0, 2).reshape(future.shape[1], -1).unsqueeze(0)
        else:
            past_fit, future_fit = past, future

        correlations, right_vectors, cholesky_past = _larimore_decomposition(
            past_fit, future_fit, ridge=self.ridge
        )
        if self.n_states > correlations.shape[-1]:
            raise ValueError("n_states exceeds identifiable subspace rank")

        if self.mode == "pooled":
            whitened_past = torch.linalg.solve_triangular(
                cholesky_past.transpose(-1, -2), past_fit, upper=True
            )
            flat_states = (
                correlations[..., : self.n_states].unsqueeze(-1)
                * right_vectors[..., : self.n_states, :]
            ) @ whitened_past
            states = flat_states.squeeze(0).T.reshape(batch, n_columns, self.n_states)
        else:
            whitened_past = torch.linalg.solve_triangular(
                cholesky_past.transpose(-1, -2), past, upper=True
            )
            state_columns = (
                correlations[..., : self.n_states].unsqueeze(-1)
                * right_vectors[..., : self.n_states, :]
            ) @ whitened_past
            states = state_columns.transpose(-1, -2)

        transition, observation, gain, innovation_covariance, innovations = (
            _fit_innovations_state_space_from_states(
                values,
                states,
                observation_start=self.past_horizon,
                mode=self.mode,
                covariance=self.covariance,
                min_covar=max(self.ridge, 1e-12),
            )
        )
        if single and self.mode == "independent":
            transition = transition[0]
            observation = observation[0]
            gain = gain[0]
            innovation_covariance = innovation_covariance[0]
        self.system_ = InnovationsStateSpace(
            transition, observation, gain, innovation_covariance
        )
        self.transition_ = transition
        self.observation_ = observation
        self.kalman_gain_ = gain
        self.innovation_covariance_ = innovation_covariance
        self.canonical_correlations_ = (
            correlations.squeeze(0) if self.mode == "pooled" else correlations
        )
        self.states_ = states[0] if single else states
        self.innovations_ = innovations[0] if single else innovations
        self.n_states_ = self.n_states
        return self


class LinearGaussianEM(BaseEstimator):
    r"""Refine linear Gaussian state-space systems by batched EM.

    ``mode="pooled"`` estimates one system from independent trajectories by
    summing sufficient statistics over batch and time. ``mode="independent"``
    estimates one system per trajectory. Trial boundaries are never used as
    state transitions.

    References
    ----------
    - Shumway, R. H. and Stoffer, D. S. (1982).
    """

    def __init__(
        self,
        system,
        n_iter: int = 20,
        min_covar: float = 1e-7,
        *,
        mode: str = "pooled",
    ):
        """Initialize EM refinement."""
        self.system = system
        self.n_iter = n_iter
        self.min_covar = min_covar
        self.mode = mode

    def fit(self, observations, y=None):
        """Run EM on one trajectory or a batch of independent trajectories."""
        del y
        if self.n_iter < 1 or self.min_covar < 0:
            raise ValueError("invalid EM settings")
        if self.mode not in {"pooled", "independent"}:
            raise ValueError("mode must be 'pooled' or 'independent'")
        values, single = _normalise_ss_observations(
            observations,
            device=self.system.transition.device,
            dtype=self.system.transition.dtype,
        )
        system = self.system
        total_history = []
        trajectory_history = []

        for _ in range(self.n_iter):
            smoothed = kalman_smoother(values, system)
            mean = smoothed.smoothed_mean
            covariance = smoothed.smoothed_covariance
            lag_covariance = smoothed.lag_covariance
            if mean.ndim == 2:
                mean = mean.unsqueeze(0)
                covariance = covariance.unsqueeze(0)
                lag_covariance = lag_covariance.unsqueeze(0)
            second = covariance + mean.unsqueeze(-1) * mean.unsqueeze(-2)
            cross = lag_covariance[:, 1:] + (
                mean[:, 1:].unsqueeze(-1) * mean[:, :-1].unsqueeze(-2)
            )

            if self.mode == "pooled":
                previous_second = second[:, :-1].sum(dim=(0, 1))
                next_second = second[:, 1:].sum(dim=(0, 1))
                cross_moment = cross.sum(dim=(0, 1))
                transition = cross_moment @ torch.linalg.pinv(previous_second)
                yz = torch.einsum("btm,btd->md", values, mean)
                observation = yz @ torch.linalg.pinv(second.sum(dim=(0, 1)))
                process_covariance = _em_process_covariance(
                    previous_second,
                    next_second,
                    cross_moment,
                    transition,
                    values.shape[0] * (values.shape[1] - 1),
                )
                residual = values - torch.einsum("md,btd->btm", observation, mean)
                observation_covariance = symmetrise(
                    (
                        torch.einsum("btm,btn->mn", residual, residual)
                        + torch.einsum(
                            "md,btdk,nk->mn", observation, covariance, observation
                        )
                    )
                    / (values.shape[0] * values.shape[1])
                )
                state_covariance = second.mean(dim=(0, 1))
            else:
                previous_second = second[:, :-1].sum(dim=1)
                next_second = second[:, 1:].sum(dim=1)
                cross_moment = cross.sum(dim=1)
                transition = cross_moment @ torch.linalg.pinv(previous_second)
                yz = torch.einsum("btm,btd->bmd", values, mean)
                observation = yz @ torch.linalg.pinv(second.sum(dim=1))
                process_covariance = _em_process_covariance(
                    previous_second,
                    next_second,
                    cross_moment,
                    transition,
                    values.shape[1] - 1,
                )
                residual = values - torch.einsum("bmd,btd->btm", observation, mean)
                observation_covariance = symmetrise(
                    (
                        torch.einsum("btm,btn->bmn", residual, residual)
                        + torch.einsum(
                            "bmd,btdk,bnk->bmn", observation, covariance, observation
                        )
                    )
                    / values.shape[1]
                )
                state_covariance = second.mean(dim=1)

            process_covariance = _covariance_floor(process_covariance, self.min_covar)
            observation_covariance = _covariance_floor(
                observation_covariance, self.min_covar
            )
            if single and self.mode == "independent":
                transition = transition[0]
                observation = observation[0]
                process_covariance = process_covariance[0]
                observation_covariance = observation_covariance[0]
                state_covariance = state_covariance[0]
            system = StateSpaceModel(
                transition,
                observation,
                process_covariance,
                observation_covariance,
                state_covariance=state_covariance,
            )
            likelihood = smoothed.log_likelihood
            if likelihood.ndim == 0:
                likelihood = likelihood.unsqueeze(0)
            trajectory_history.append(likelihood.detach().clone())
            total_history.append(float(likelihood.sum()))

        self.system_ = system
        self.log_likelihood_history_ = (
            torch.stack(trajectory_history)
            if self.mode == "independent"
            else total_history
        )
        self.trajectory_log_likelihood_history_ = torch.stack(trajectory_history)
        return self


def _normalise_ss_observations(observations, *, device, dtype):
    """Normalize estimator input to batch-first floating-point observations."""
    source = torch.as_tensor(observations)
    target_device = source.device if device == "auto" else torch.device(device)
    values = source.to(device=target_device, dtype=_resolve_dtype(dtype))
    single = values.ndim == 2
    if single:
        values = values.unsqueeze(0)
    if values.ndim != 3:
        raise ValueError("observations must have shape (time,n) or (batch,time,n)")
    if not torch.isfinite(values).all():
        raise ValueError("observations must be finite")
    return values, single


def _batch_covariance(samples):
    """Return one unbiased covariance matrix per batch."""
    centered = samples - samples.mean(dim=1, keepdim=True)
    denominator = max(1, samples.shape[1] - 1)
    return symmetrise(centered.transpose(-1, -2) @ centered / denominator)


def _pooled_covariance(samples):
    """Return covariance after pooling samples without creating transitions."""
    flat = samples.reshape(-1, samples.shape[-1])
    centered = flat - flat.mean(dim=0, keepdim=True)
    return symmetrise(centered.T @ centered / max(1, flat.shape[0] - 1))


def _fit_general_state_space_from_states(values, states, *, observation_start, mode, min_covar):
    """Estimate ``A,C,Q,R`` while respecting trajectory boundaries."""
    previous = states[:, :-1]
    following = states[:, 1:]
    observations = values[:, observation_start : observation_start + previous.shape[1]]
    if mode == "pooled":
        x0 = previous.reshape(-1, previous.shape[-1])
        x1 = following.reshape(-1, following.shape[-1])
        y0 = observations.reshape(-1, observations.shape[-1])
        transition = torch.linalg.lstsq(x0, x1).solution.T
        observation = torch.linalg.lstsq(x0, y0).solution.T
        process_residual = x1 - x0 @ transition.T
        observation_residual = y0 - x0 @ observation.T
        process_covariance = _covariance_floor(_batch_covariance(process_residual.unsqueeze(0))[0], min_covar)
        observation_covariance = _covariance_floor(_batch_covariance(observation_residual.unsqueeze(0))[0], min_covar)
    else:
        transition = torch.linalg.lstsq(previous, following).solution.transpose(-1, -2)
        observation = torch.linalg.lstsq(previous, observations).solution.transpose(-1, -2)
        process_residual = following - previous @ transition.transpose(-1, -2)
        observation_residual = observations - previous @ observation.transpose(-1, -2)
        process_covariance = _covariance_floor(_batch_covariance(process_residual), min_covar)
        observation_covariance = _covariance_floor(_batch_covariance(observation_residual), min_covar)
    return transition, observation, process_covariance, observation_covariance



def _fit_innovations_state_space_from_states(values, states, *, observation_start, mode, covariance, min_covar):
    """Estimate ``A,C,K,V`` from Larimore state estimates.

    ``C`` and ``V`` use every available state/observation pair, matching the
    CVA regression step ``y_t = C x_t + e_t``. ``A`` and ``K`` use only valid
    within-trajectory transitions ``(x_t, x_{t+1})``; trial boundaries are
    never treated as state transitions.
    """
    previous = states[:, :-1]
    following = states[:, 1:]
    observations_all = values[
        :, observation_start : observation_start + states.shape[1]
    ]
    denominator_adjustment = 1 if covariance == "unbiased" else 0
    if mode == "pooled":
        x_all = states.reshape(-1, states.shape[-1])
        y_all = observations_all.reshape(-1, observations_all.shape[-1])
        x0 = previous.reshape(-1, previous.shape[-1])
        x1 = following.reshape(-1, following.shape[-1])

        # Larimore step 2: regress every available y_t on its estimated state.
        observation = torch.linalg.lstsq(x_all, y_all).solution.T
        innovations_flat = y_all - x_all @ observation.T
        innovations = innovations_flat.reshape(
            values.shape[0], states.shape[1], values.shape[-1]
        )

        # Larimore step 3: regress only genuine within-trial state transitions.
        transition = torch.linalg.lstsq(x0, x1).solution.T
        state_residual = x1 - x0 @ transition.T
        transition_innovations = innovations[:, :-1].reshape(
            -1, innovations.shape[-1]
        )
        gain = torch.linalg.lstsq(
            transition_innovations, state_residual
        ).solution.T

        denominator = max(
            1, innovations_flat.shape[0] - denominator_adjustment
        )
        innovation_covariance = symmetrise(
            innovations_flat.T @ innovations_flat / denominator
        )
    else:
        observation = torch.linalg.lstsq(
            states, observations_all
        ).solution.transpose(-1, -2)
        innovations = observations_all - states @ observation.transpose(-1, -2)

        transition = torch.linalg.lstsq(
            previous, following
        ).solution.transpose(-1, -2)
        state_residual = following - previous @ transition.transpose(-1, -2)
        transition_innovations = innovations[:, :-1]
        gain = torch.linalg.lstsq(
            transition_innovations, state_residual
        ).solution.transpose(-1, -2)

        denominator = max(
            1, innovations.shape[1] - denominator_adjustment
        )
        innovation_covariance = symmetrise(
            innovations.transpose(-1, -2) @ innovations / denominator
        )
    innovation_covariance = _covariance_floor(innovation_covariance, min_covar)
    return transition, observation, gain, innovation_covariance, innovations


def _em_process_covariance(previous, following, cross, transition, denominator):
    """Evaluate the closed-form EM process-noise covariance update."""
    return symmetrise(
        (
            following
            - transition @ cross.transpose(-1, -2)
            - cross @ transition.transpose(-1, -2)
            + transition @ previous @ transition.transpose(-1, -2)
        )
        / denominator
    )


def _covariance_floor(covariance, floor):
    """Symmetrise covariance and add a diagonal numerical floor."""
    identity = torch.eye(
        covariance.shape[-1], dtype=covariance.dtype, device=covariance.device
    )
    return symmetrise(covariance) + floor * identity
