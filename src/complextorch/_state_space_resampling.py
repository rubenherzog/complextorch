r"""Private state-space resampling mechanics for confidence intervals.

All supported state-space estimators use the same observable-process bootstrap.
A fitted model is converted once to the canonical steady-state innovations form,
innovation vectors are resampled or drawn, trajectories are simulated without
crossing boundaries, and the same fixed-complexity estimator is refitted.
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
    """Normalize observations through the estimator's dtype/device contract."""
    if isinstance(estimator, LinearGaussianEM):
        return _normalise_ss_observations(
            x,
            device=estimator.system.transition.device,
            dtype=estimator.system.transition.dtype,
        )[0]
    return _normalise_ss_observations(x, device=estimator.device, dtype=estimator.dtype)[0]


def _fitted_canonical_system(estimator: StateSpaceEstimator) -> CanonicalStateSpace:
    """Return the canonical system exposed by a fitted state-space estimator."""
    if not hasattr(estimator, "system_"):
        raise RuntimeError("state-space estimator must be fitted before resampling")
    return estimator.system_


def _state_space_prediction_innovations(
    system: CanonicalStateSpace,
    trials: torch.Tensor,
) -> torch.Tensor:
    """Return steady-state one-step innovations for independent trajectories."""
    canonical = as_innovations_state_space(system)
    transition = canonical.transition
    observation = canonical.observation
    gain = canonical.gain
    if transition.ndim == 2:
        transition = transition.unsqueeze(0)
        observation = observation.unsqueeze(0)
        gain = gain.unsqueeze(0)
    batch = trials.shape[0]
    matrices = (transition, observation, gain)
    if transition.shape[0] == 1:
        transition, observation, gain = [
            value.expand(batch, *value.shape[1:]) for value in matrices
        ]
    elif transition.shape[0] != batch:
        raise ValueError("fitted state-space batch does not match trajectories")

    state = torch.zeros(
        (batch, transition.shape[-1]), dtype=trials.dtype, device=trials.device
    )
    errors = torch.empty_like(trials)
    for time in range(trials.shape[1]):
        error = trials[:, time] - torch.einsum("bij,bj->bi", observation, state)
        errors[:, time] = error
        state = (
            torch.einsum("bij,bj->bi", transition, state)
            + torch.einsum("bij,bj->bi", gain, error)
        )
    return errors


def _simulate_state_space_resamples(
    system: CanonicalStateSpace,
    innovations: torch.Tensor,
) -> torch.Tensor:
    """Simulate observations from canonical innovations-form dynamics."""
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
        transition, observation, gain = [
            value.expand(batch, *value.shape[1:])
            for value in (transition, observation, gain)
        ]
    elif transition.shape[0] != batch:
        raise ValueError("fitted state-space batch does not match trajectories")

    state = torch.zeros(
        (n_resamples, batch, transition.shape[-1]),
        dtype=innovations.dtype,
        device=innovations.device,
    )
    observations = torch.empty_like(innovations)
    for time in range(length):
        error = innovations[:, :, time]
        observations[:, :, time] = (
            torch.einsum("bij,rbj->rbi", observation, state) + error
        )
        state = (
            torch.einsum("bij,rbj->rbi", transition, state)
            + torch.einsum("bij,rbj->rbi", gain, error)
        )
    return observations


def _stack_state_space_systems(systems: list[CanonicalStateSpace]) -> CanonicalStateSpace:
    """Stack systems, flattening an existing estimator batch when necessary."""
    if not systems:
        raise ValueError("at least one system is required")
    first = systems[0]

    def combine(values: list[torch.Tensor]) -> torch.Tensor:
        """Combine unbatched systems by stacking and batched systems by concatenation."""
        return torch.stack(values) if values[0].ndim == 2 else torch.cat(values, dim=0)

    if isinstance(first, InnovationsStateSpace):
        if not all(isinstance(system, InnovationsStateSpace) for system in systems):
            raise TypeError("cannot stack mixed state-space representations")
        return InnovationsStateSpace(
            combine([system.transition for system in systems]),
            combine([system.observation for system in systems]),
            combine([system.gain for system in systems]),
            combine([system.innovation_covariance for system in systems]),
        )
    if not all(isinstance(system, StateSpaceModel) for system in systems):
        raise TypeError("cannot stack mixed state-space representations")
    state_covariance = None
    if all(system.state_covariance is not None for system in systems):
        state_covariance = combine([system.state_covariance for system in systems])
    return StateSpaceModel(
        combine([system.transition for system in systems]),
        combine([system.observation for system in systems]),
        combine([system.process_covariance for system in systems]),
        combine([system.observation_covariance for system in systems]),
        state_covariance=state_covariance,
    )


def _prepare_state_space_refit(
    estimator: StateSpaceEstimator,
    original_system: CanonicalStateSpace,
    mode: str,
) -> StateSpaceEstimator:
    """Clone one fixed-complexity estimator for a bootstrap refit."""
    refit = copy.deepcopy(estimator)
    refit.mode = mode
    if isinstance(refit, LinearGaussianEM):
        if not isinstance(original_system, StateSpaceModel):
            raise TypeError("LinearGaussianEM requires a general StateSpaceModel")
        refit.system = original_system
    return refit


def _refit_state_space_resamples(
    samples: torch.Tensor,
    estimator: StateSpaceEstimator,
    original_system: CanonicalStateSpace,
) -> CanonicalStateSpace:
    """Refit surrogates while preserving resample and trajectory semantics."""
    n_resamples, batch, time, n_variables = samples.shape

    # N4SID and Larimore support a batched independent fit directly. EM also
    # does when there is only one original initialization to broadcast.
    can_flatten = not isinstance(estimator, LinearGaussianEM) or batch == 1
    if (estimator.mode == "independent" or batch == 1) and can_flatten:
        flat = samples.reshape(n_resamples * batch, time, n_variables)
        refit = _prepare_state_space_refit(estimator, original_system, "independent")
        return _fitted_canonical_system(refit.fit(flat))

    # Pooled bootstrap has two logical batch axes, while estimators expose one.
    # Independent EM with multiple original systems likewise needs the original
    # trajectory-specific initialization preserved. Loop only over resamples.
    systems: list[CanonicalStateSpace] = []
    for index in range(n_resamples):
        refit = _prepare_state_space_refit(estimator, original_system, estimator.mode)
        systems.append(_fitted_canonical_system(refit.fit(samples[index])))
    return _stack_state_space_systems(systems)


def _state_space_stable_mask(
    system: CanonicalStateSpace,
    *,
    n_resamples: int,
    n_trials: int,
    mode: str,
) -> torch.Tensor:
    """Return resamples whose state transition is stable in every trajectory."""
    transition = system.transition
    if transition.ndim == 2:
        transition = transition.unsqueeze(0)
    stable = torch.linalg.eigvals(transition).abs().amax(-1) < 1
    if mode == "pooled":
        return stable
    return stable.reshape(n_resamples, n_trials).all(dim=1)


def _select_state_space_ensemble(
    system: CanonicalStateSpace,
    valid: torch.Tensor,
    *,
    n_resamples: int,
    n_trials: int,
    mode: str,
) -> CanonicalStateSpace:
    """Select stable systems while preserving a common bootstrap sample axis."""
    def select(value: torch.Tensor | None) -> torch.Tensor | None:
        """Select accepted resamples from one optional canonical matrix."""
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
