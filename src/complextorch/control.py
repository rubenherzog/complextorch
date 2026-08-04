"""Control-theoretic linear algebra for state-space inference and reduction."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from scipy.linalg import solve_discrete_are

from .linalg import spd_logdet, symmetrise
from .representations import LinearDynamicalSystem


def _batched(t: torch.Tensor, ndim: int) -> tuple[torch.Tensor, bool]:
    x = torch.as_tensor(t)
    single = x.ndim == ndim - 1
    return (x.unsqueeze(0) if single else x), single


def solve_dare(transition, observation, process_covariance, observation_covariance):
    """Solve the filtering discrete algebraic Riccati equation."""
    a, single = _batched(transition, 3); c, _ = _batched(observation, 3)
    q, _ = _batched(process_covariance, 3); r, _ = _batched(observation_covariance, 3)
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
    p = solve_dare(system.transition, system.observation, system.process_covariance, system.observation_covariance)
    a, single = _batched(system.transition, 3); c, _ = _batched(system.observation, 3); r, _ = _batched(system.observation_covariance, 3)
    if p.ndim == 2: p = p.unsqueeze(0)
    batch=max(a.shape[0],c.shape[0],r.shape[0],p.shape[0]); a,c,r,p=[x.expand(batch,*x.shape[1:]) if x.shape[0]==1 else x for x in (a,c,r,p)]
    v=symmetrise(c@p@c.transpose(-1,-2)+r); gain=a@p@c.transpose(-1,-2)@torch.linalg.inv(v)
    return InnovationsForm(v[0] if single else v,gain[0] if single else gain,p[0] if single else p)


def reduce_state_space(system: LinearDynamicalSystem, indices):
    idx=torch.as_tensor(indices,dtype=torch.long,device=system.observation.device)
    c=system.observation.index_select(-2,idx); r=system.observation_covariance.index_select(-2,idx).index_select(-1,idx)
    names=None if system.channel_names is None else tuple(system.channel_names[i] for i in idx.tolist())
    return LinearDynamicalSystem(system.transition,c,system.process_covariance,r,system.state_covariance,system.sampling_frequency,names)


def dynamical_dependence(system: LinearDynamicalSystem, *, base: float = 2.0):
    if system.state_covariance is None: raise ValueError("state_covariance is required")
    sy=symmetrise(system.observation@system.state_covariance@system.observation.transpose(-1,-2)+system.observation_covariance)
    v=innovations_form(system).covariance
    return .5*(spd_logdet(sy)-spd_logdet(v))/np.log(base)


def stochastic_interaction(system: LinearDynamicalSystem, groups, *, base: float = 2.0):
    parts=torch.stack([dynamical_dependence(reduce_state_space(system,g),base=base) for g in groups],-1)
    return parts.sum(-1)-dynamical_dependence(system,base=base)
