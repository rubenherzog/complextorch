"""Control-theoretic linear algebra for state-space inference and reduction."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from scipy.linalg import solve_discrete_are
from .linalg import spd_logdet, spd_solve, symmetrise
from .representations import LinearDynamicalSystem


def _batched(t: torch.Tensor, ndim: int) -> tuple[torch.Tensor, bool]:
    x = torch.as_tensor(t)
    single = x.ndim == ndim - 1
    return (x.unsqueeze(0) if single else x), single


def solve_dare(transition, observation, process_covariance, observation_covariance):
    """Solve the filtering discrete algebraic Riccati equation."""
    a, single = _batched(transition, 3)
    c, _ = _batched(observation, 3)
    q, _ = _batched(process_covariance, 3)
    r, _ = _batched(observation_covariance, 3)
    batch = max(a.shape[0], c.shape[0], q.shape[0], r.shape[0])
    tensors = [x.expand(batch, *x.shape[1:]) if x.shape[0] == 1 else x for x in (a, c, q, r)]
    out = []
    for ai, ci, qi, ri in zip(*[x.detach().cpu().numpy() for x in tensors], strict=True):
        out.append(torch.as_tensor(solve_discrete_are(ai.T, ci.T, qi, ri), dtype=a.dtype, device=a.device))
    result = symmetrise(torch.stack(out))
    return result[0] if single else result


@dataclass(frozen=True)
class InnovationsForm:
    covariance: torch.Tensor
    gain: torch.Tensor
    prediction_covariance: torch.Tensor


def innovations_form(system: LinearDynamicalSystem) -> InnovationsForm:
    """Return steady-state innovations covariance and Kalman gain."""
    p = solve_dare(system.transition, system.observation, system.process_covariance, system.observation_covariance)
    a, single = _batched(system.transition, 3)
    c, _ = _batched(system.observation, 3)
    r, _ = _batched(system.observation_covariance, 3)
    if p.ndim == 2:
        p = p.unsqueeze(0)
    batch = max(a.shape[0], c.shape[0], r.shape[0], p.shape[0])
    a, c, r, p = [x.expand(batch, *x.shape[1:]) if x.shape[0] == 1 else x for x in (a, c, r, p)]
    innovation_covariance = symmetrise(c @ p @ c.transpose(-1, -2) + r)
    identity = torch.eye(innovation_covariance.shape[-1], dtype=innovation_covariance.dtype, device=innovation_covariance.device).expand_as(innovation_covariance)
    gain = a @ p @ c.transpose(-1, -2) @ spd_solve(innovation_covariance, identity)
    return InnovationsForm(innovation_covariance[0] if single else innovation_covariance, gain[0] if single else gain, p[0] if single else p)


def reduce_state_space(system: LinearDynamicalSystem, indices) -> LinearDynamicalSystem:
    """Marginalise observations while preserving latent dynamics."""
    index = torch.as_tensor(indices, dtype=torch.long, device=system.observation.device)
    observation = system.observation.index_select(-2, index)
    noise = system.observation_covariance.index_select(-2, index).index_select(-1, index)
    names = None if system.channel_names is None else tuple(system.channel_names[i] for i in index.tolist())
    return LinearDynamicalSystem(system.transition, observation, system.process_covariance, noise, system.state_covariance, system.sampling_frequency, names)


def project_state_space(system: LinearDynamicalSystem, projection: torch.Tensor) -> LinearDynamicalSystem:
    """Apply a linear observation projection while sharing latent dynamics."""
    matrix = torch.as_tensor(projection, dtype=system.observation.dtype, device=system.observation.device)
    if matrix.ndim == 2 and system.observation.ndim == 3:
        matrix = matrix.unsqueeze(0)
    observation = matrix @ system.observation
    noise = symmetrise(matrix @ system.observation_covariance @ matrix.transpose(-1, -2))
    return LinearDynamicalSystem(system.transition, observation, system.process_covariance, noise, system.state_covariance, system.sampling_frequency, None)


def dynamical_dependence(system: LinearDynamicalSystem, *, base: float = 2.0):
    if system.state_covariance is None:
        raise ValueError("state_covariance is required")
    stationary = symmetrise(system.observation @ system.state_covariance @ system.observation.transpose(-1, -2) + system.observation_covariance)
    innovations = innovations_form(system).covariance
    return 0.5 * (spd_logdet(stationary) - spd_logdet(innovations)) / np.log(base)


def stochastic_interaction(system: LinearDynamicalSystem, groups, *, base: float = 2.0):
    """SSDI/stochastic interaction using the common reduced-model path."""
    parts = torch.stack([dynamical_dependence(reduce_state_space(system, group), base=base) for group in groups], -1)
    return parts.sum(-1) - dynamical_dependence(system, base=base)


@dataclass(frozen=True)
class ProjectionSearchResult:
    projection: torch.Tensor
    objective: torch.Tensor
    history: torch.Tensor


def optimise_dynamical_dependence_projection(system: LinearDynamicalSystem, output_dimension: int, *, n_candidates: int = 256, seed: int = 0, minimise: bool = True) -> ProjectionSearchResult:
    """Reproducible Stiefel search reusing projection, DARE and DD primitives."""
    n_observations = system.observation.shape[-2]
    if not 1 <= output_dimension <= n_observations:
        raise ValueError("output_dimension must be between 1 and observation dimension")
    generator = torch.Generator(device=system.observation.device).manual_seed(seed)
    values, projections = [], []
    for _ in range(n_candidates):
        raw = torch.randn((n_observations, output_dimension), generator=generator, dtype=system.observation.dtype, device=system.observation.device)
        orthogonal, _ = torch.linalg.qr(raw, mode="reduced")
        projection = orthogonal.transpose(-1, -2)
        values.append(dynamical_dependence(project_state_space(system, projection)))
        projections.append(projection)
    history = torch.stack(values)
    scores = history.mean(tuple(range(1, history.ndim))) if history.ndim > 1 else history
    index = torch.argmin(scores) if minimise else torch.argmax(scores)
    return ProjectionSearchResult(projections[int(index)], history[index], history)
