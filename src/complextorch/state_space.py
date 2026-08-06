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
from .representations import LinearDynamicalSystem


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
    """Estimate a linear state-space model by compact N4SID.

    The algorithm forms block-Hankel past and future matrices, projects the
    future onto the past row space, and truncates the resulting SVD to the
    requested latent dimension.

    References
    ----------
    - Van Overschee and De Moor (1994).
    """

    def __init__(
        self,
        n_states: int,
        block_rows: int = 10,
        ridge: float = 1e-8,
    ):
        """Initialize the N4SID estimator.

        Parameters
        ----------
        n_states
            Latent state dimension retained from the singular-value
            decomposition.
        block_rows
            Number of block rows in past and future Hankel matrices.
        ridge
            Diagonal regularization applied to the past Gram matrix.
        """

        self.n_states = n_states
        self.block_rows = block_rows
        self.ridge = ridge

    def fit(self, observations):
        """Fit transition, observation and noise covariances.

        Parameters
        ----------
        observations
            Single trajectory with shape ``(time, variables)``.

        Returns
        -------
        N4SID
            Fitted estimator with ``system_``, ``singular_values_`` and
            ``states_`` attributes.
        """

        values = torch.as_tensor(observations, dtype=torch.float64)
        if values.ndim != 2:
            raise ValueError("N4SID currently expects (time,variables)")
        n_times, n_variables = values.shape
        block_rows = self.block_rows
        if n_times <= 2 * block_rows + 2:
            raise ValueError("not enough samples for requested block_rows")

        n_columns = n_times - 2 * block_rows + 1
        hankel = torch.stack(
            [values[offset : offset + n_columns].T for offset in range(2 * block_rows)],
            dim=0,
        )
        past = hankel[:block_rows].reshape(
            block_rows * n_variables, n_columns
        )
        future = hankel[block_rows:].reshape(
            block_rows * n_variables, n_columns
        )

        # Project future blocks onto the regularized row space of past blocks.
        regularized_gram = past @ past.T + self.ridge * torch.eye(
            block_rows * n_variables, dtype=values.dtype
        )
        projection = future @ past.T @ torch.linalg.pinv(
            regularized_gram
        ) @ past
        left, singular_values, _ = torch.linalg.svd(
            projection, full_matrices=False
        )
        observability = left[:, : self.n_states] @ torch.diag(
            torch.sqrt(singular_values[: self.n_states])
        )
        states = torch.linalg.lstsq(observability, future).solution.T

        previous_states = states[:-1]
        next_states = states[1:]
        aligned_observations = values[
            block_rows : block_rows + states.shape[0] - 1
        ]
        transition = torch.linalg.lstsq(
            previous_states, next_states
        ).solution.T
        observation = torch.linalg.lstsq(
            previous_states, aligned_observations
        ).solution.T
        process_residual = next_states - previous_states @ transition.T
        observation_residual = (
            aligned_observations - previous_states @ observation.T
        )
        process_covariance = symmetrise(
            process_residual.T
            @ process_residual
            / max(1, process_residual.shape[0] - 1)
        )
        observation_covariance = symmetrise(
            observation_residual.T
            @ observation_residual
            / max(1, observation_residual.shape[0] - 1)
        )

        self.system_ = LinearDynamicalSystem(
            transition,
            observation,
            process_covariance,
            observation_covariance,
            state_covariance=torch.eye(self.n_states, dtype=values.dtype),
        )
        self.singular_values_ = singular_values
        self.states_ = states
        return self


class LinearGaussianEM(BaseEstimator):
    """Refine a linear Gaussian state-space model by EM.

    The E-step uses RTS-smoothed first and second moments. The M-step updates
    ``A``, ``C``, ``Q`` and ``R`` by their closed-form Gaussian maximum-
    likelihood expressions.

    References
    ----------
    - Shumway and Stoffer (1982).
    """

    def __init__(
        self,
        system,
        n_iter: int = 20,
        min_covar: float = 1e-7,
    ):
        """Initialize EM refinement.

        Parameters
        ----------
        system
            Initial linear Gaussian state-space model.
        n_iter
            Number of EM iterations.
        min_covar
            Diagonal floor added to process and observation covariances.
        """

        self.system = system
        self.n_iter = n_iter
        self.min_covar = min_covar

    def fit(self, observations):
        """Run EM on one observed trajectory.

        Parameters
        ----------
        observations
            Observations with shape ``(time, variables)``.

        Returns
        -------
        LinearGaussianEM
            Fitted estimator with ``system_`` and
            ``log_likelihood_history_`` attributes.
        """

        values = torch.as_tensor(
            observations,
            dtype=self.system.transition.dtype,
            device=self.system.transition.device,
        )
        system = self.system
        history = []

        for _ in range(self.n_iter):
            smoothed = kalman_smoother(values, system)
            mean = smoothed.smoothed_mean
            covariance = smoothed.smoothed_covariance
            lag_covariance = smoothed.lag_covariance
            second_moment = covariance + mean.unsqueeze(-1) * mean.unsqueeze(-2)

            previous_second = second_moment[:-1].sum(0)
            next_second = second_moment[1:].sum(0)
            cross_moment = (
                lag_covariance[1:]
                + mean[1:].unsqueeze(-1) * mean[:-1].unsqueeze(-2)
            ).sum(0)

            # Closed-form M-step regressions from expected sufficient statistics.
            transition = cross_moment @ torch.linalg.pinv(previous_second)
            observation = (
                values.T @ mean
            ) @ torch.linalg.pinv(second_moment.sum(0))
            process_covariance = symmetrise(
                (
                    next_second
                    - transition @ cross_moment.T
                    - cross_moment @ transition.T
                    + transition @ previous_second @ transition.T
                )
                / (mean.shape[0] - 1)
            )
            residual = values - mean @ observation.T
            observation_covariance = symmetrise(
                (
                    residual.T @ residual
                    + torch.einsum(
                        "ij,tjk,lk->il",
                        observation,
                        covariance,
                        observation,
                    )
                )
                / mean.shape[0]
            )

            # A covariance floor prevents singular M-step estimates.
            process_covariance = process_covariance + self.min_covar * torch.eye(
                process_covariance.shape[-1],
                dtype=process_covariance.dtype,
                device=process_covariance.device,
            )
            observation_covariance = (
                observation_covariance
                + self.min_covar
                * torch.eye(
                    observation_covariance.shape[-1],
                    dtype=observation_covariance.dtype,
                    device=observation_covariance.device,
                )
            )
            system = LinearDynamicalSystem(
                transition,
                observation,
                process_covariance,
                observation_covariance,
                state_covariance=second_moment.mean(0),
            )
            history.append(float(smoothed.log_likelihood))

        self.system_ = system
        self.log_likelihood_history_ = history
        return self
