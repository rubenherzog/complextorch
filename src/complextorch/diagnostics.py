"""Fit, predictive, and recovery diagnostics for linear dynamical estimators.

The diagnostics deliberately separate predictive accuracy from innovations-model
adequacy. All adequacy statistics operate on one-step-ahead errors and share the
same covariance, standardisation, and multivariate-whiteness calculations across
VAR and linear state-space estimators. State-space recovery comparisons use only
quantities that are invariant to invertible changes of latent coordinates.

References
----------
- Ding, M., Bressler, S. L., Yang, W., and Liang, H. (2000). Short-window
  spectral analysis of cortical event-related potentials by adaptive
  multivariate autoregressive modeling. *Biological Cybernetics*, 83, 35--45.
- Hosking, J. R. M. (1980). The multivariate portmanteau statistic. *Journal of
  the American Statistical Association*, 75, 602--608.
- Ljung, G. M. and Box, G. E. P. (1978). On a measure of lack of fit in time
  series models. *Biometrika*, 65, 297--303.
- Kailath, T. (1980). *Linear Systems*. Prentice-Hall.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .control import InnovationsStateSpace, _as_innovations_state_space
from .linalg import stable_cholesky
from .representations import StateSpaceModel
from .state_space import LarimoreStateSpace, LinearGaussianEM, N4SID
from .var import VAR


@dataclass(frozen=True)
class FitDiagnostics:
    """Prediction and innovations diagnostics.

    Scalar statistics are scalars for a single trajectory or ``mode='pooled'``.
    For batched ``mode='independent'`` they have shape ``(batch,)``. Matrix
    diagnostics preserve the corresponding leading batch dimension.

    ``prediction_mean`` and ``standardized_errors`` retain the trajectory layout
    of the aligned observations. ``prediction_covariance`` is the covariance
    assumed by the fitted one-step predictor, whereas
    ``innovation_covariance_oos`` is the empirical covariance of the observed
    prediction errors. ``prediction_interval_coverage`` is the empirical
    marginal coverage of the nominal 95% Gaussian predictive interval.
    """

    rmse: torch.Tensor
    nmse: torch.Tensor
    predictive_r2: torch.Tensor
    gaussian_nll: torch.Tensor
    consistency: torch.Tensor
    innovation_covariance_oos: torch.Tensor
    standardized_mean: torch.Tensor
    standardized_covariance: torch.Tensor
    covariance_calibration: torch.Tensor
    autocorrelation_matrices: torch.Tensor
    whiteness_energy: torch.Tensor
    portmanteau_statistic: torch.Tensor
    max_abs_autocorrelation: torch.Tensor
    durbin_watson: torch.Tensor
    n_observations: torch.Tensor
    prediction_mean: torch.Tensor | None = None
    prediction_covariance: torch.Tensor | None = None
    standardized_errors: torch.Tensor | None = None
    prediction_interval_coverage: torch.Tensor | None = None


@dataclass(frozen=True)
class StateSpaceRecoveryDiagnostics:
    """Gauge-invariant comparison between two linear state-space systems.

    A latent change of coordinates ``x -> T x`` changes ``A``, ``C``, and the
    latent noise matrices but leaves the observation process unchanged. Recovery
    is therefore summarized using transition eigenvalue sets, innovations
    Markov/Hankel blocks, their singular values, and the observation innovation
    covariance rather than entrywise latent-parameter errors.

    Scalar errors are scalars for unbatched systems and vectors for batched
    systems. A singleton system may be broadcast against a batched system.
    """

    transition_eigenvalues: torch.Tensor
    reference_transition_eigenvalues: torch.Tensor
    spectral_distance: torch.Tensor
    hankel_singular_values: torch.Tensor
    reference_hankel_singular_values: torch.Tensor
    hankel_relative_error: torch.Tensor
    hankel_spectrum_relative_error: torch.Tensor
    innovation_covariance_relative_error: torch.Tensor
    horizon: int


def _normalise_observations(values, *, dtype=None, device=None) -> tuple[torch.Tensor, bool]:
    """Normalize observations to ``(batch,time,variables)`` without reshaping data."""
    data = torch.as_tensor(values, dtype=dtype, device=device)
    single = data.ndim == 2
    if single:
        data = data.unsqueeze(0)
    if data.ndim != 3:
        raise ValueError("observations must have shape (time,n) or (batch,time,n)")
    if data.shape[1] < 1 or data.shape[2] < 1:
        raise ValueError("observations must contain time points and variables")
    if not bool(torch.isfinite(data).all()):
        raise ValueError("observations must contain only finite values")
    return data, single


def _expand_matrix(matrix: torch.Tensor, batch: int) -> torch.Tensor:
    """Broadcast one fitted matrix over observation trajectories."""
    value = torch.as_tensor(matrix)
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3:
        raise ValueError("expected a fitted matrix or batched fitted matrices")
    if value.shape[0] == 1 and batch > 1:
        value = value.expand(batch, *value.shape[1:])
    if value.shape[0] != batch:
        raise ValueError("fitted model batch dimension does not match observations")
    return value


def _expand_vector(vector: torch.Tensor, batch: int) -> torch.Tensor:
    """Broadcast one fitted vector over observation trajectories."""
    value = torch.as_tensor(vector)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2:
        raise ValueError("expected a fitted vector or batched fitted vectors")
    if value.shape[0] == 1 and batch > 1:
        value = value.expand(batch, -1)
    if value.shape[0] != batch:
        raise ValueError("fitted model batch dimension does not match observations")
    return value


def _var_oos_errors(estimator: VAR, train: torch.Tensor, test: torch.Tensor) -> torch.Tensor:
    """Return rolling one-step VAR errors on an unseen contiguous test block."""
    if train.shape[1] < estimator.order:
        raise ValueError("training observations are shorter than the fitted VAR order")
    coefficients = torch.as_tensor(estimator.coef_).to(train)
    if coefficients.shape[0] == 1 and train.shape[0] > 1:
        coefficients = coefficients.expand(train.shape[0], *coefficients.shape[1:])
    if coefficients.shape[0] != train.shape[0]:
        raise ValueError("fitted model batch dimension does not match observations")
    intercept = _expand_vector(estimator.intercept_, train.shape[0]).to(train)
    history = train[:, -estimator.order :].clone()
    errors = []
    for time_index in range(test.shape[1]):
        prediction = intercept.clone()
        for lag in range(estimator.order):
            prediction = prediction + torch.einsum(
                "bij,bj->bi", coefficients[:, lag], history[:, -(lag + 1)]
            )
        error = test[:, time_index] - prediction
        errors.append(error)
        history = torch.cat((history[:, 1:], test[:, time_index : time_index + 1]), 1)
    return torch.stack(errors, 1)


def _ss_oos_errors(estimator, train: torch.Tensor, test: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return rolling one-step state-space errors and per-trajectory training means."""
    system = _as_innovations_state_space(estimator.system_)
    batch = train.shape[0]
    transition = _expand_matrix(system.transition, batch).to(train)
    observation = _expand_matrix(system.observation, batch).to(train)
    gain = _expand_matrix(system.gain, batch).to(train)
    fitted_mean = getattr(estimator, "mean_", None)
    if fitted_mean is None:
        training_mean = train.mean(dim=1, keepdim=True)
    else:
        training_mean = torch.as_tensor(fitted_mean, dtype=train.dtype, device=train.device)
        if training_mean.ndim == 1:
            training_mean = training_mean[None, None]
        elif training_mean.ndim == 2:
            training_mean = training_mean[:, None]
        if training_mean.shape[0] == 1 and batch > 1:
            training_mean = training_mean.expand(batch, -1, -1)
        if training_mean.shape != (batch, 1, train.shape[-1]):
            raise ValueError("fitted centering mean does not match observations")
    centered_train = train - training_mean
    centered_test = test - training_mean
    state = torch.zeros(
        batch,
        transition.shape[-1],
        dtype=train.dtype,
        device=train.device,
    )
    for time_index in range(centered_train.shape[1]):
        innovation = centered_train[:, time_index] - torch.einsum(
            "bmd,bd->bm", observation, state
        )
        state = torch.einsum("bij,bj->bi", transition, state) + torch.einsum(
            "bdm,bm->bd", gain, innovation
        )
    errors = []
    for time_index in range(centered_test.shape[1]):
        innovation = centered_test[:, time_index] - torch.einsum(
            "bmd,bd->bm", observation, state
        )
        errors.append(innovation)
        state = torch.einsum("bij,bj->bi", transition, state) + torch.einsum(
            "bdm,bm->bd", gain, innovation
        )
    return torch.stack(errors, 1), training_mean


def _innovation_covariance(estimator, batch: int, reference: torch.Tensor) -> torch.Tensor:
    """Return the fitted innovation covariance expanded over trajectories."""
    if isinstance(estimator, VAR):
        covariance = estimator.noise_covariance_
    else:
        covariance = _as_innovations_state_space(estimator.system_).innovation_covariance
    return _expand_matrix(covariance, batch).to(reference)


def _covariance(values: torch.Tensor, *, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample covariance with pooled or independent trajectory semantics."""
    if values.shape[1] < 2:
        raise ValueError("at least two test observations are required")
    centered = values - values.mean(dim=1, keepdim=True)
    if mode == "pooled":
        numerator = torch.einsum("btn,btm->nm", centered, centered)
        count = values.shape[0] * values.shape[1]
        degrees = count - values.shape[0]
        if degrees < 1:
            raise ValueError("not enough pooled observations to estimate covariance")
        return numerator / float(degrees), torch.tensor(count, device=values.device)
    numerator = centered.transpose(-1, -2) @ centered
    count = values.shape[1]
    counts = torch.full((values.shape[0],), count, dtype=torch.long, device=values.device)
    return numerator / float(count - 1), counts


def _prediction_consistency(
    observations: torch.Tensor,
    errors: torch.Tensor,
    *,
    training_mean: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Ding--Bressler covariance-structure consistency on aligned OOS samples."""
    centered = observations - training_mean
    prediction = centered - errors
    if mode == "pooled":
        target = centered.reshape(-1, centered.shape[-1])
        predicted = prediction.reshape(-1, prediction.shape[-1])
        rr = target.T @ target / float(target.shape[0] - 1)
        rs = predicted.T @ predicted / float(target.shape[0] - 1)
        return 1.0 - torch.linalg.matrix_norm(rs - rr) / torch.linalg.matrix_norm(rr)
    rr = centered.transpose(-1, -2) @ centered / float(centered.shape[1] - 1)
    rs = prediction.transpose(-1, -2) @ prediction / float(centered.shape[1] - 1)
    return 1.0 - torch.linalg.matrix_norm(rs - rr, dim=(-2, -1)) / torch.linalg.matrix_norm(
        rr, dim=(-2, -1)
    )


def _standardize(errors: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
    """Apply the fitted :math:`V^{-1/2}` to every one-step prediction error."""
    chol, _ = stable_cholesky(covariance, jitter=1e-10)
    return torch.linalg.solve_triangular(
        chol[:, None], errors.unsqueeze(-1), upper=False
    ).squeeze(-1)


def _lag_autocorrelations(
    standardized: torch.Tensor,
    *,
    max_lag: int,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return sample-normalized lag correlations, whiteness energy, and Ljung--Box Q."""
    if max_lag < 1 or max_lag >= standardized.shape[1]:
        raise ValueError("max_lag must satisfy 1 <= max_lag < n_test")
    sample_covariance, counts = _covariance(standardized, mode=mode)
    covariance = sample_covariance.unsqueeze(0) if sample_covariance.ndim == 2 else sample_covariance
    chol, _ = stable_cholesky(covariance, jitter=1e-10)
    lag_values = []
    q_terms = []
    for lag in range(1, max_lag + 1):
        left = standardized[:, lag:] - standardized[:, lag:].mean(dim=1, keepdim=True)
        right = standardized[:, :-lag] - standardized[:, :-lag].mean(dim=1, keepdim=True)
        if mode == "pooled":
            cross = torch.einsum("btn,btm->nm", left, right) / float(
                standardized.shape[0] * (standardized.shape[1] - lag)
            )
            cross = cross.unsqueeze(0)
        else:
            cross = (left.transpose(-1, -2) @ right) / float(standardized.shape[1] - lag)
        first = torch.linalg.solve_triangular(chol, cross, upper=False)
        correlation = torch.linalg.solve_triangular(
            chol, first.transpose(-1, -2), upper=False
        ).transpose(-1, -2)
        lag_values.append(correlation)
        energy = correlation.square().sum(dim=(-2, -1))
        valid_pairs = (
            standardized.shape[0] * (standardized.shape[1] - lag)
            if mode == "pooled"
            else standardized.shape[1] - lag
        )
        q_terms.append(energy / float(valid_pairs))
    correlations = torch.stack(lag_values, dim=-3)
    energy = correlations.square().sum(dim=(-3, -2, -1))
    n_eff = (
        standardized.shape[0] * standardized.shape[1]
        if mode == "pooled"
        else standardized.shape[1]
    )
    q = float(n_eff * (n_eff + 2)) * torch.stack(q_terms, -1).sum(-1)
    if mode == "pooled":
        return correlations[0], energy[0], q[0]
    return correlations, energy, q


def _durbin_watson(errors: torch.Tensor, *, mode: str) -> torch.Tensor:
    """Boundary-safe per-variable Durbin--Watson statistic."""
    numerator = (errors[:, 1:] - errors[:, :-1]).square().sum(dim=1)
    denominator = errors.square().sum(dim=1)
    if mode == "pooled":
        return numerator.sum(dim=0) / denominator.sum(dim=0)
    return numerator / denominator


def innovation_diagnostics(
    observations,
    errors,
    innovation_covariance,
    *,
    training_mean,
    max_lag: int = 10,
    mode: str = "pooled",
) -> FitDiagnostics:
    """Compute common diagnostics from aligned one-step prediction errors.

    This lower-level function is model-family agnostic. Independent trajectories
    are never linked when forming lagged residual products.
    """
    if mode not in {"pooled", "independent"}:
        raise ValueError("mode must be 'pooled' or 'independent'")
    data, single = _normalise_observations(observations)
    residual, _ = _normalise_observations(errors, dtype=data.dtype, device=data.device)
    if residual.shape != data.shape:
        raise ValueError("errors must have the same shape as aligned observations")
    mean = torch.as_tensor(training_mean, dtype=data.dtype, device=data.device)
    if mean.ndim == 1:
        mean = mean[None, None]
    elif mean.ndim == 2:
        mean = mean[:, None]
    if mean.shape[0] == 1 and data.shape[0] > 1:
        mean = mean.expand(data.shape[0], -1, -1)
    if mean.shape != (data.shape[0], 1, data.shape[-1]):
        raise ValueError("training_mean must match the observation batch and variables")
    covariance = _expand_matrix(torch.as_tensor(innovation_covariance), data.shape[0]).to(data)
    standardized = _standardize(residual, covariance)
    observed_covariance, counts = _covariance(residual, mode=mode)
    standardized_covariance, _ = _covariance(standardized, mode=mode)
    std_cov_batch = standardized_covariance.unsqueeze(0) if standardized_covariance.ndim == 2 else standardized_covariance
    identity = torch.eye(data.shape[-1], dtype=data.dtype, device=data.device)
    calibration = torch.linalg.matrix_norm(std_cov_batch - identity, dim=(-2, -1)) / math.sqrt(
        data.shape[-1]
    )
    standardized_mean = standardized.mean(dim=1)
    if mode == "pooled":
        standardized_mean = standardized.reshape(-1, data.shape[-1]).mean(0)
        calibration = calibration[0]
    autocorrelation, whiteness_energy, q = _lag_autocorrelations(
        standardized, max_lag=max_lag, mode=mode
    )
    squared_error = residual.square().sum(dim=(-2, -1))
    baseline = (data - mean).square().sum(dim=(-2, -1))
    if mode == "pooled":
        squared_error = squared_error.sum()
        baseline = baseline.sum()
        rmse = torch.sqrt(residual.square().mean())
    else:
        rmse = torch.sqrt(residual.square().mean(dim=(-2, -1)))
    nmse = squared_error / baseline
    r2 = 1.0 - nmse
    chol, _ = stable_cholesky(covariance, jitter=1e-10)
    solved = torch.cholesky_solve(residual.unsqueeze(-1), chol[:, None]).squeeze(-1)
    nll_each = 0.5 * (
        (residual * solved).sum(-1)
        + 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)[:, None]
        + residual.shape[-1] * math.log(2.0 * math.pi)
    )
    nll = nll_each.mean() if mode == "pooled" else nll_each.mean(1)
    consistency = _prediction_consistency(
        data, residual, training_mean=mean, mode=mode
    )
    max_abs = autocorrelation.abs().amax(dim=(-3, -2, -1)) if mode == "independent" else autocorrelation.abs().amax()
    dw = _durbin_watson(residual, mode=mode)

    prediction = data - residual
    marginal_scale = torch.sqrt(torch.diagonal(covariance, dim1=-2, dim2=-1)).clamp_min(
        torch.finfo(data.dtype).eps
    )
    within_95 = residual.abs() <= 1.959963984540054 * marginal_scale[:, None]
    coverage = within_95.to(data.dtype).mean(dim=(-2, -1))
    prediction_covariance = covariance
    if mode == "pooled":
        coverage = within_95.to(data.dtype).mean()
        prediction_covariance = covariance[0]

    if single:
        prediction = prediction[0]
        standardized_out = standardized[0]
    else:
        standardized_out = standardized
    if single and mode == "independent":
        def squeeze(value):
            """Remove the singleton batch axis from one diagnostic output."""
            return value[0] if value.ndim > 0 and value.shape[0] == 1 else value
        observed_covariance = squeeze(observed_covariance)
        standardized_covariance = squeeze(standardized_covariance)
        standardized_mean = squeeze(standardized_mean)
        autocorrelation = squeeze(autocorrelation)
        rmse, nmse, r2, nll, consistency = map(squeeze, (rmse, nmse, r2, nll, consistency))
        calibration, whiteness_energy, q, max_abs, dw, counts = map(
            squeeze, (calibration, whiteness_energy, q, max_abs, dw, counts)
        )
        prediction_covariance = squeeze(prediction_covariance)
        coverage = squeeze(coverage)
    return FitDiagnostics(
        rmse=rmse,
        nmse=nmse,
        predictive_r2=r2,
        gaussian_nll=nll,
        consistency=consistency,
        innovation_covariance_oos=observed_covariance,
        standardized_mean=standardized_mean,
        standardized_covariance=standardized_covariance,
        covariance_calibration=calibration,
        autocorrelation_matrices=autocorrelation,
        whiteness_energy=whiteness_energy,
        portmanteau_statistic=q,
        max_abs_autocorrelation=max_abs,
        durbin_watson=dw,
        n_observations=counts,
        prediction_mean=prediction,
        prediction_covariance=prediction_covariance,
        standardized_errors=standardized_out,
        prediction_interval_coverage=coverage,
    )


def _fit_diagnostics_oos(
    estimator,
    train,
    test,
    *,
    max_lag: int = 10,
) -> FitDiagnostics:
    """Evaluate a fitted VAR or linear state-space model on unseen observations.

    The test block is consumed in rolling one-step-ahead mode: each observation
    is scored before it is used to update the predictor. ``train`` is used only
    to initialize the prediction recursion and the training-only centering
    baseline; the estimator is never refitted by this function.
    """
    train_data, train_single = _normalise_observations(train)
    test_data, test_single = _normalise_observations(
        test, dtype=train_data.dtype, device=train_data.device
    )
    if train_single != test_single or train_data.shape[0] != test_data.shape[0]:
        raise ValueError("train and test must have matching batch semantics")
    if train_data.shape[-1] != test_data.shape[-1]:
        raise ValueError("train and test must have the same number of variables")
    mode = getattr(estimator, "mode", "pooled")
    if isinstance(estimator, VAR):
        if not hasattr(estimator, "coef_"):
            raise ValueError("estimator must be fitted before diagnostics")
        errors = _var_oos_errors(estimator, train_data, test_data)
        training_mean = train_data.mean(dim=1, keepdim=True)
    elif isinstance(estimator, (LarimoreStateSpace, N4SID, LinearGaussianEM)):
        if not hasattr(estimator, "system_"):
            raise ValueError("estimator must be fitted before diagnostics")
        errors, training_mean = _ss_oos_errors(estimator, train_data, test_data)
    else:
        raise TypeError(
            "estimator must be a fitted VAR, LarimoreStateSpace, N4SID, or LinearGaussianEM"
        )
    covariance = _innovation_covariance(estimator, test_data.shape[0], test_data)
    return innovation_diagnostics(
        test_data[0] if test_single else test_data,
        errors[0] if test_single else errors,
        covariance[0] if test_single else covariance,
        training_mean=training_mean[0, 0] if test_single else training_mean[:, 0],
        max_lag=max_lag,
        mode=mode,
    )


def _ss_in_sample_errors(estimator, data: torch.Tensor):
    """Return aligned in-sample innovations from the fitted innovations recursion."""
    system = _as_innovations_state_space(estimator.system_)
    batch = data.shape[0]
    transition = _expand_matrix(system.transition, batch).to(data)
    observation = _expand_matrix(system.observation, batch).to(data)
    gain = _expand_matrix(system.gain, batch).to(data)
    fitted_mean = getattr(estimator, "mean_", None)
    if fitted_mean is None:
        training_mean = data.mean(dim=1, keepdim=True)
    else:
        training_mean = torch.as_tensor(fitted_mean, dtype=data.dtype, device=data.device)
        if training_mean.ndim == 1:
            training_mean = training_mean[None, None]
        elif training_mean.ndim == 2:
            training_mean = training_mean[:, None]
        if training_mean.shape[0] == 1 and batch > 1:
            training_mean = training_mean.expand(batch, -1, -1)
        if training_mean.shape != (batch, 1, data.shape[-1]):
            raise ValueError("fitted centering mean does not match observations")
    centered = data - training_mean
    if isinstance(estimator, LarimoreStateSpace):
        warmup = int(estimator.past_horizon)
    elif isinstance(estimator, N4SID):
        warmup = int(estimator.block_rows)
    else:
        warmup = 0
    if warmup >= data.shape[1]:
        raise ValueError(
            "state-space identification horizon leaves no diagnostic samples"
        )
    state = torch.zeros(
        batch, transition.shape[-1], dtype=data.dtype, device=data.device
    )
    errors = []
    aligned = []
    for time_index in range(centered.shape[1]):
        innovation = centered[:, time_index] - torch.einsum(
            "bmd,bd->bm", observation, state
        )
        if time_index >= warmup:
            errors.append(innovation)
            aligned.append(data[:, time_index])
        state = torch.einsum("bij,bj->bi", transition, state) + torch.einsum(
            "bdm,bm->bd", gain, innovation
        )
    return torch.stack(aligned, 1), torch.stack(errors, 1), training_mean


def fit_diagnostics(
    estimator,
    train,
    test=None,
    *,
    evaluation: str = "oos",
    max_lag: int = 10,
) -> FitDiagnostics:
    """Evaluate fitted VAR/state-space models in- or out-of-sample.

    ``evaluation="oos"`` preserves the rolling held-out contract.
    ``evaluation="in_sample"`` computes one-step residual diagnostics on the
    supplied fitted-data trajectory without refitting the estimator. Both modes
    delegate all statistics to :func:`innovation_diagnostics`, so predictive
    outputs and innovations diagnostics share exactly the same recursion.
    """
    if evaluation not in {"oos", "in_sample"}:
        raise ValueError("evaluation must be 'oos' or 'in_sample'")
    if evaluation == "oos":
        if test is None:
            raise ValueError("test is required for evaluation='oos'")
        return _fit_diagnostics_oos(estimator, train, test, max_lag=max_lag)
    if test is not None:
        raise ValueError("test must be omitted for evaluation='in_sample'")

    data, single = _normalise_observations(train)
    mode = getattr(estimator, "mode", "pooled")
    training_mean = data.mean(dim=1, keepdim=True)
    if isinstance(estimator, VAR):
        if not hasattr(estimator, "coef_"):
            raise ValueError("estimator must be fitted before diagnostics")
        if data.shape[1] <= estimator.order:
            raise ValueError("observations are too short for the fitted VAR order")
        errors = estimator.residuals(data).to(data)
        aligned = data[:, estimator.order :]
    elif isinstance(estimator, (LarimoreStateSpace, N4SID, LinearGaussianEM)):
        if not hasattr(estimator, "system_"):
            raise ValueError("estimator must be fitted before diagnostics")
        aligned, errors, training_mean = _ss_in_sample_errors(estimator, data)
    else:
        raise TypeError(
            "estimator must be a fitted VAR, LarimoreStateSpace, N4SID, or LinearGaussianEM"
        )

    covariance = _innovation_covariance(estimator, aligned.shape[0], aligned)
    return innovation_diagnostics(
        aligned[0] if single else aligned,
        errors[0] if single else errors,
        covariance[0] if single else covariance,
        training_mean=training_mean[0, 0] if single else training_mean[:, 0],
        max_lag=max_lag,
        mode=mode,
    )


def _extract_state_space_system(value):
    """Return a supported state-space representation from a model or estimator."""
    system = value.system_ if hasattr(value, "system_") else value
    if not isinstance(system, (StateSpaceModel, InnovationsStateSpace)):
        raise TypeError(
            "expected a StateSpaceModel, InnovationsStateSpace, or fitted state-space estimator"
        )
    return system


def _broadcast_innovations_systems(estimate, reference):
    """Convert two systems to innovations form and broadcast their batch axes."""
    first = _as_innovations_state_space(_extract_state_space_system(estimate))
    second = _as_innovations_state_space(_extract_state_space_system(reference))
    matrices_first = [first.transition, first.observation, first.gain, first.innovation_covariance]
    matrices_second = [second.transition, second.observation, second.gain, second.innovation_covariance]
    first_single = all(torch.as_tensor(value).ndim == 2 for value in matrices_first)
    second_single = all(torch.as_tensor(value).ndim == 2 for value in matrices_second)
    matrices_first = [torch.as_tensor(value).unsqueeze(0) if torch.as_tensor(value).ndim == 2 else torch.as_tensor(value) for value in matrices_first]
    matrices_second = [torch.as_tensor(value).unsqueeze(0) if torch.as_tensor(value).ndim == 2 else torch.as_tensor(value) for value in matrices_second]
    if matrices_first[1].shape[-2] != matrices_second[1].shape[-2]:
        raise ValueError("state-space systems must have the same observation dimension")
    dtype = torch.promote_types(matrices_first[0].dtype, matrices_second[0].dtype)
    device = matrices_first[0].device
    if matrices_second[0].device != device:
        raise ValueError("state-space systems must be on the same device")
    matrices_first = [value.to(dtype=dtype) for value in matrices_first]
    matrices_second = [value.to(dtype=dtype) for value in matrices_second]
    batch = max(matrices_first[0].shape[0], matrices_second[0].shape[0])
    if matrices_first[0].shape[0] not in {1, batch} or matrices_second[0].shape[0] not in {1, batch}:
        raise ValueError("incompatible state-space batch dimensions")
    matrices_first = [value.expand(batch, *value.shape[1:]) if value.shape[0] == 1 else value for value in matrices_first]
    matrices_second = [value.expand(batch, *value.shape[1:]) if value.shape[0] == 1 else value for value in matrices_second]
    return matrices_first, matrices_second, first_single and second_single


def _innovations_hankel(transition, observation, gain, covariance, horizon):
    """Build a finite innovations Hankel matrix from gauge-invariant Markov blocks."""
    chol, _ = stable_cholesky(covariance, jitter=1e-10)
    state = torch.eye(
        transition.shape[-1], dtype=transition.dtype, device=transition.device
    ).expand(transition.shape[0], transition.shape[-1], transition.shape[-1])
    blocks = []
    for _ in range(2 * horizon - 1):
        blocks.append(observation @ state @ gain @ chol)
        state = state @ transition
    rows = [torch.cat(blocks[index : index + horizon], dim=-1) for index in range(horizon)]
    return torch.cat(rows, dim=-2)


def _relative_matrix_error(estimate, reference):
    """Return Frobenius error normalized by the reference matrix norm."""
    denominator = torch.linalg.matrix_norm(reference, ord="fro", dim=(-2, -1)).clamp_min(
        torch.finfo(reference.dtype).eps
    )
    return torch.linalg.matrix_norm(estimate - reference, ord="fro", dim=(-2, -1)) / denominator


def _spectral_set_distance(estimate, reference):
    """Symmetric nearest-neighbour distance between unordered eigenvalue sets."""
    distance = (estimate.unsqueeze(-1) - reference.unsqueeze(-2)).abs()
    forward = distance.amin(-1).mean(-1)
    backward = distance.amin(-2).mean(-1)
    scale = reference.abs().mean(-1).clamp_min(torch.finfo(reference.real.dtype).eps)
    return 0.5 * (forward + backward) / scale


def state_space_recovery_diagnostics(
    estimate,
    reference,
    *,
    horizon: int = 5,
) -> StateSpaceRecoveryDiagnostics:
    """Compare two state-space models using latent-basis-invariant quantities.

    Parameters
    ----------
    estimate, reference
        ``StateSpaceModel`` or ``InnovationsStateSpace`` instances, or fitted
        state-space estimators exposing ``system_``. General state-space models
        are converted to steady-state innovations form before comparison.
    horizon
        Number of block rows and columns in the finite innovations Hankel
        matrix. No data trajectories are joined or reshaped by this comparison.

    Returns
    -------
    StateSpaceRecoveryDiagnostics
        Gauge-invariant spectral, Hankel, and innovation-covariance errors.
    """
    if int(horizon) != horizon or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    horizon = int(horizon)
    first, second, single = _broadcast_innovations_systems(estimate, reference)
    a1, c1, k1, v1 = first
    a2, c2, k2, v2 = second
    eig1 = torch.linalg.eigvals(a1)
    eig2 = torch.linalg.eigvals(a2)
    hankel1 = _innovations_hankel(a1, c1, k1, v1, horizon)
    hankel2 = _innovations_hankel(a2, c2, k2, v2, horizon)
    singular1 = torch.linalg.svdvals(hankel1)
    singular2 = torch.linalg.svdvals(hankel2)
    spectral_distance = _spectral_set_distance(eig1, eig2)
    hankel_error = _relative_matrix_error(hankel1, hankel2)
    spectrum_denominator = torch.linalg.vector_norm(singular2, dim=-1).clamp_min(
        torch.finfo(singular2.dtype).eps
    )
    spectrum_error = torch.linalg.vector_norm(singular1 - singular2, dim=-1) / spectrum_denominator
    covariance_error = _relative_matrix_error(v1, v2)
    if single:
        eig1, eig2 = eig1[0], eig2[0]
        singular1, singular2 = singular1[0], singular2[0]
        spectral_distance = spectral_distance[0]
        hankel_error = hankel_error[0]
        spectrum_error = spectrum_error[0]
        covariance_error = covariance_error[0]
    return StateSpaceRecoveryDiagnostics(
        transition_eigenvalues=eig1,
        reference_transition_eigenvalues=eig2,
        spectral_distance=spectral_distance,
        hankel_singular_values=singular1,
        reference_hankel_singular_values=singular2,
        hankel_relative_error=hankel_error,
        hankel_spectrum_relative_error=spectrum_error,
        innovation_covariance_relative_error=covariance_error,
        horizon=horizon,
    )


__all__ = [
    "FitDiagnostics",
    "StateSpaceRecoveryDiagnostics",
    "fit_diagnostics",
    "innovation_diagnostics",
    "state_space_recovery_diagnostics",
]
