r"""Private state-space resampling mechanics for confidence intervals.

General :class:`~complextorch.StateSpaceModel` and
:class:`~complextorch.InnovationsStateSpace` fits share one observable-process
bootstrap: convert exactly to steady-state innovations form, obtain or draw
innovation vectors, simulate without crossing trajectory boundaries, refit the
same fixed-complexity estimator, and retain stable systems. Measure evaluation
remains separate in :mod:`complextorch.inference_registry`.

References
----------
- Shumway, R. H. and Stoffer, D. S. (1982). An approach to time series
  smoothing and forecasting using the EM algorithm.
- Van Overschee, P. and De Moor, B. (1994). N4SID.
- Larimore, W. E. (1990, 1996). Canonical variate analysis for system
  identification.
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from .control import InnovationsStateSpace
from .representations import StateSpaceModel
from .state_space import (
    LarimoreStateSpace,
    LinearGaussianEM,
    N4SID,
    _normalise_ss_observations,
)
from .transformations import as_innovations_state_space

StateSpaceEstimator = N4SID | LarimoreStateSpace | LinearGaussianEM
CanonicalStateSpace = StateSpaceModel | InnovationsStateSpace


def _normalise_state_space_trials(
    estimator: StateSpaceEstimator,
    x: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """Normalize observations through the state-space estimator input contract."""
    if isinstance(estimator, LinearGaussianEM):
        values, _ = _normalise_ss_observations(
            x,
            device=estimator.system.transition.device,
            dtype=estimator.system.transition.dtype,
        )
        return values
    values, _ = _normalise_ss_observations(
        x, device=estimator.device, dtype=estimator.dtype
    )
    return values


def _fitted_canonical_system(estimator: StateSpaceEstimator) -> CanonicalStateSpace:
    """Return the canonical fitted system from a state-space estimator."""
    if not hasattr(estimator, "system_"):
        raise RuntimeError("state-space estimator must be fitted before resampling")
    return estimator.system_


def _state_space_prediction_innovations(
    system: CanonicalStateSpace,
    trials: torch.Tensor,
) -> torch.Tensor:
    """Compute one-step innovations from a canonical state-space process."""
    innovations = as_innovations_state_space(system)
    transition = innovations.transition
    observation = innovations.observation
    gain = innovations.gain
    if transition.ndim == 2:
        transition = transition.unsqueeze(0)
        observation = observation.unsqueeze(0)
        gain = gain.unsqueeze(0)
    batch = trials.shape[0]
    if transition.shape[0] == 1:
        transition = transition.expand(batch, -1, -1)
        observation = observation.expand(batch, -1, -1)
        gain = gain.expand(batch, -1, -1)
    elif transition.shape[0] != batch:
        raise ValueError("fitted state-space batch does not match trajectories")
    state = torch.zeros(
        (batch, transition.shape[-1]), dtype=trials.dtype, device=trials.device
    )
    output = torch.empty_like(trials)
    for time in range(trials.shape[1]):
        error = trials[:, time] - torch.einsum("bij,bj->bi", observation, state)
        output[:, time] = error
        state = (
            torch.einsum("bij,bj->bi", transition, state)
            + torch.einsum("bij,bj->bi", gain, error)
        )
    return output


def _simulate_state_space_resamples(
    system: CanonicalStateSpace,
    innovations: torch.Tensor,
) -> torch.Tensor:
    """Simulate observable trajectories from shared innovations-form dynamics."""
    canonical = as_innovations_state_space(system)
    transition = canonical.transition
    observation = canonical.observation
    gain = canonical.gain
    if transition.ndim == 2:
        transition = transition.unsqueeze(0)
        observation = observation.unsqueeze(0)
        gain = gain.unsqueeze(0)
    n_resamples, batch, length, _ = innovations.shape
    if transition.shape[0] == 1:
        transition = transition.expand(batch, -1, -1)
        observation = observation.expand(batch, -1, -1)
        gain = gain.expand(batch, -1, -1)
    elif transition.shape[0] != batch:
        raise ValueError("fitted state-space batch does not match trajectories")
    state = torch.zeros(
        (n_resamples, batch, transition.shape[-1]),
        dtype=innovations.dtype,
        device=innovations.device,
    )
    output = torch.empty_like(innovations)
    for time in range(length):
        error = innovations[:, :, time]
        output[:, :, time] = (
            torch.einsum("bij,rbj->rbi", observation, state) + error
        )
        state = (
            torch.einsum("bij,rbj->rbi", transition, state)
            + torch.einsum("bij,rbj->rbi", gain, error)
        )
    return output


def _stack_state_space_systems(
    systems: list[CanonicalStateSpace],
) -> CanonicalStateSpace:
    """Stack homogeneous canonical state-space systems along one batch axis."""
    if not systems:
        raise ValueError("at least one system is required")
    first = systems[0]
    if isinstance(first, InnovationsStateSpace):
        if not all(isinstance(system, InnovationsStateSpace) for system in systems):
            raise TypeError("cannot stack mixed state-space representations")
        return InnovationsStateSpace(
            torch.stack([system.transition for system in systems]),
            torch.stack([system.observation for system in systems]),
            torch.stack([system.gain for system in systems]),
            torch.stack([system.innovation_covariance for system in systems]),
        )
    if not all(isinstance(system, StateSpaceModel) for system in systems):
        raise TypeError("cannot stack mixed state-space representations")
    state_covariance = None
    if all(system.state_covariance is not None for system in systems):
        state_covariance = torch.stack([system.state_covariance for system in systems])
    return StateSpaceModel(
        torch.stack([system.transition for system in systems]),
        torch.stack([system.observation for system in systems]),
        torch.stack([system.process_covariance for system in systems]),
        torch.stack([system.observation_covariance for system in systems]),
        state_covariance=state_covariance,
    )


def _prepare_state_space_refit(
    estimator: StateSpaceEstimator,
    original_system: CanonicalStateSpace,
    mode: str,
) -> StateSpaceEstimator:
    """Clone a fixed-complexity estimator for one bootstrap refit stage."""
    refit = copy.deepcopy(estimator)
    refit.mode = mode
    if isinstance(refit, LinearGaussianEM):
        refit.system = original_system
    return refit


def _refit_state_space_resamples(
    samples: torch.Tensor,
    estimator: StateSpaceEstimator,
    original_system: CanonicalStateSpace,
) -> CanonicalStateSpace:
    """Refit fixed-complexity state-space models without crossing trajectories."""
    n_resamples, batch, time, n_variables = samples.shape
    if estimator.mode == "independent" or batch == 1:
        flat = samples.reshape(n_resamples * batch, time, n_variables)
        refit = _prepare_state_space_refit(estimator, original_system, "independent")
        return _fitted_canonical_system(refit.fit(flat))

    # State-space estimators currently expose one batch axis. A pooled resample
    # has two logical axes (resample, trajectory), so loop only over resamples
    # while preserving the trajectory batch inside each pooled fit.
    systems: list[CanonicalStateSpace] = []
    for index in range(n_resamples):
        refit = _prepare_state_space_refit(estimator, original_system, "pooled")
        systems.append(_fitted_canonical_system(refit.fit(samples[index])))
    return _stack_state_space_systems(systems)


def _state_space_stable_mask(
    system: CanonicalStateSpace,
    *,
    n_resamples: int,
    n_trials: int,
    mode: str,
) -> torch.Tensor:
    """Return bootstrap replicates whose fitted state transition is stable."""
    transition = system.transition
    if transition.ndim == 2:
        transition = transition.unsqueeze(0)
    radius = torch.linalg.eigvals(transition).abs().amax(-1)
    if mode == "pooled":
        return radius < 1
    return (radius.reshape(n_resamples, n_trials) < 1).all(dim=1)


def _select_state_space_ensemble(
    system: CanonicalStateSpace,
    valid: torch.Tensor,
    *,
    n_resamples: int,
    n_trials: int,
    mode: str,
) -> CanonicalStateSpace:
    """Select valid state-space replicates while retaining a common sample axis."""

    def select(value: torch.Tensor | None) -> torch.Tensor | None:
        """Select valid replicate rows from one canonical system matrix."""
        if value is None:
            return None
        tensor = value if value.ndim == 3 else value.unsqueeze(0)
        if mode == "pooled":
            return tensor[valid]
        shaped = tensor.reshape(n_resamples, n_trials, *tensor.shape[1:])
        return shaped[valid].reshape(-1, *tensor.shape[1:])

    if isinstance(system, InnovationsStateSpace):
        return InnovationsStateSpace(
            select(system.transition),
            select(system.observation),
            select(system.gain),
            select(system.innovation_covariance),
        )
    return StateSpaceModel(
        select(system.transition),
        select(system.observation),
        select(system.process_covariance),
        select(system.observation_covariance),
        state_covariance=select(system.state_covariance),
    )
