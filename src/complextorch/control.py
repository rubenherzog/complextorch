"""Control-theoretic transformations for linear Gaussian systems.

For

.. math::

   z_{t+1}=Az_t+w_t,\qquad y_t=Cz_t+v_t,

steady-state Kalman and innovations quantities are obtained from discrete
algebraic Riccati equations, including process--observation cross covariance
when required.

References
----------
- Kalman, R. E. (1960). A new approach to linear filtering and prediction.
- Anderson, B. D. O. and Moore, J. B. (1979). *Optimal Filtering*.
- Barnett, L. and Seth, A. K. (2015). Granger causality for state-space models.
- Barnett, L. and Seth, A. K. (2023). Dynamical independence: Discovering
  emergent macroscopic processes in complex dynamical systems.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from scipy.linalg import solve_discrete_are
from .linalg import spd_logdet, spd_solve, symmetrise
from .representations import StateSpaceModel, VARSystem


def _batched(t: torch.Tensor, ndim: int) -> tuple[torch.Tensor, bool]:
    """Normalize one tensor to a leading batch dimension."""
    x = torch.as_tensor(t)
    single = x.ndim == ndim - 1
    return (x.unsqueeze(0) if single else x), single


def _validate_log_base(base: float) -> float:
    """Return a valid logarithm base as a Python float."""
    value = float(base)
    if not np.isfinite(value) or value <= 0.0 or value == 1.0:
        raise ValueError("base must be finite, positive, and different from 1")
    return value


def solve_dare(transition, observation, process_covariance, observation_covariance):
    """Solve the steady-state discrete algebraic Riccati equation.

    References
    ----------
    - Anderson and Moore (1979), *Optimal Filtering*.
    """
    # Iterate the steady-state Kalman covariance recursion until the Riccati fixed point is reached.
    a, single = _batched(transition, 3)
    c, _ = _batched(observation, 3)
    q, _ = _batched(process_covariance, 3)
    r, _ = _batched(observation_covariance, 3)
    batch = max(a.shape[0], c.shape[0], q.shape[0], r.shape[0])
    tensors = [
        x.expand(batch, *x.shape[1:]) if x.shape[0] == 1 else x
        for x in (a, c, q, r)
    ]
    out = []
    for ai, ci, qi, ri in zip(
        *[x.detach().cpu().numpy() for x in tensors], strict=True
    ):
        out.append(
            torch.as_tensor(
                solve_discrete_are(ai.T, ci.T, qi, ri),
                dtype=a.dtype,
                device=a.device,
            )
        )
    result = symmetrise(torch.stack(out))
    return result[0] if single else result


# Marginal innovations require the steady-state generalised Riccati solution.
def solve_generalized_dare(
    transition: torch.Tensor,
    observation: torch.Tensor,
    process_covariance: torch.Tensor,
    observation_covariance: torch.Tensor,
    cross_covariance: torch.Tensor,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_iter: int = 10000,
) -> torch.Tensor:
    """Solve the generalized DARE with noise cross covariance.

    References
    ----------
    - Anderson and Moore (1979); Barnett and Seth (2015).
    """
    # Iterate the generalized Riccati map, including process-observation noise cross covariance.
    a, single = _batched(transition, 3)
    c, _ = _batched(observation, 3)
    q, _ = _batched(process_covariance, 3)
    r, _ = _batched(observation_covariance, 3)
    s, _ = _batched(cross_covariance, 3)
    batch = max(x.shape[0] for x in (a, c, q, r, s))
    a, c, q, r, s = [
        x.expand(batch, *x.shape[1:]) if x.shape[0] == 1 else x
        for x in (a, c, q, r, s)
    ]
    p = torch.zeros_like(q)
    for _ in range(max_iter):
        innovation = symmetrise(c @ p @ c.transpose(-1, -2) + r)
        gain_numerator = a @ p @ c.transpose(-1, -2) + s
        updated = symmetrise(
            a @ p @ a.transpose(-1, -2)
            + q
            - gain_numerator
            @ spd_solve(innovation, gain_numerator.transpose(-1, -2))
        )
        difference = torch.linalg.matrix_norm(
            updated - p, ord="fro", dim=(-2, -1)
        )
        scale = torch.linalg.matrix_norm(
            updated, ord="fro", dim=(-2, -1)
        ).clamp_min(1.0)
        if bool(torch.all(difference <= atol + rtol * scale)):
            return updated[0] if single else updated
        p = updated
    raise RuntimeError("generalized DARE did not converge")


@dataclass(frozen=True)
class InnovationsForm:
    """Steady-state innovations covariance, gain, and prediction covariance."""

    covariance: torch.Tensor
    gain: torch.Tensor
    prediction_covariance: torch.Tensor


def innovations_form(system: StateSpaceModel) -> InnovationsForm:
    """Convert a linear Gaussian model to steady-state innovations form.

    References
    ----------
    - Kalman (1960); Anderson and Moore (1979); Barnett and Seth (2015).
    """
    # Convert the general state-space model to predictor form using the steady-state innovation covariance and Kalman gain.
    p = solve_dare(
        system.transition,
        system.observation,
        system.process_covariance,
        system.observation_covariance,
    )
    a, single = _batched(system.transition, 3)
    c, _ = _batched(system.observation, 3)
    r, _ = _batched(system.observation_covariance, 3)
    if p.ndim == 2:
        p = p.unsqueeze(0)
    batch = max(a.shape[0], c.shape[0], r.shape[0], p.shape[0])
    a, c, r, p = [
        x.expand(batch, *x.shape[1:]) if x.shape[0] == 1 else x
        for x in (a, c, r, p)
    ]
    innovation_covariance = symmetrise(
        c @ p @ c.transpose(-1, -2) + r
    )
    identity = torch.eye(
        innovation_covariance.shape[-1],
        dtype=innovation_covariance.dtype,
        device=innovation_covariance.device,
    ).expand_as(innovation_covariance)
    gain = (
        a
        @ p
        @ c.transpose(-1, -2)
        @ spd_solve(innovation_covariance, identity)
    )
    return InnovationsForm(
        innovation_covariance[0] if single else innovation_covariance,
        gain[0] if single else gain,
        p[0] if single else p,
    )


@dataclass(frozen=True)
class InnovationsStateSpace:
    """Innovations representation x[t+1]=A x[t]+K e[t], y[t]=C x[t]+e[t]."""

    transition: torch.Tensor
    observation: torch.Tensor
    gain: torch.Tensor
    innovation_covariance: torch.Tensor


def var_to_innovations_state_space(system: VARSystem) -> InnovationsStateSpace:
    """Convert a VAR(p) exactly to companion innovations state space.

    References
    ----------
    - Lütkepohl (2005); Barnett and Seth (2015).
    """
    # Embed VAR(p) coefficients in companion form so the observation process is represented exactly in innovations form.
    coefficients = system.coefficients
    batch, order, n_variables, _ = coefficients.shape
    state_dimension = order * n_variables
    observation = coefficients.permute(0, 2, 1, 3).reshape(
        batch, n_variables, state_dimension
    )
    gain = torch.zeros(
        (batch, state_dimension, n_variables),
        dtype=coefficients.dtype,
        device=coefficients.device,
    )
    gain[:, :n_variables, :] = torch.eye(
        n_variables, dtype=coefficients.dtype, device=coefficients.device
    )
    return InnovationsStateSpace(
        system.companion,
        observation,
        gain,
        system.innovation_covariance,
    )


def _as_innovations_state_space(
    system: StateSpaceModel | InnovationsStateSpace | VARSystem,
) -> InnovationsStateSpace:
    """Return one canonical innovations representation without refitting data."""
    if isinstance(system, InnovationsStateSpace):
        return system
    if isinstance(system, VARSystem):
        return var_to_innovations_state_space(system)
    if isinstance(system, StateSpaceModel):
        form = innovations_form(system)
        return InnovationsStateSpace(
            system.transition,
            system.observation,
            form.gain,
            form.covariance,
        )
    raise TypeError(
        "system must be a StateSpaceModel, InnovationsStateSpace, or VARSystem"
    )


def _project_innovations_state_space(
    system: InnovationsStateSpace,
    projection: torch.Tensor,
) -> InnovationsStateSpace:
    r"""Return the exact innovations model of ``Y_t=L X_t``.

    For

    .. math::

       z_{t+1}=Az_t+K\varepsilon_t,\qquad
       X_t=Cz_t+\varepsilon_t,\qquad
       \operatorname{cov}(\varepsilon_t)=\Sigma,

    projection by a full-row-rank matrix :math:`L` produces

    .. math::

       C_R=LC,\qquad
       Q=K\Sigma K^\top,\qquad
       R=L\Sigma L^\top,\qquad
       S=K\Sigma L^\top.

    The generalized DARE yields the reduced prediction covariance :math:`P_R`,
    reduced innovations covariance
    :math:`\Sigma_R=C_RP_RC_R^\top+R`, and corresponding reduced innovations
    gain.  This is the single canonical implementation used by marginal
    reduction, dynamical dependence, and projected predictive measures.

    Parameters
    ----------
    system
        Microscopic innovations-form model.
    projection
        Matrix with shape ``(m,n)`` or ``(batch,m,n)``. Batch dimensions may
        broadcast only when one side has batch size one.

    Returns
    -------
    InnovationsStateSpace
        Exact reduced innovations representation, preserving dtype/device and
        returning an unbatched model only when both inputs were unbatched.

    References
    ----------
    - Barnett and Seth (2015), state-space marginalisation.
    - Barnett and Seth (2023), Eqs. (33)-(34).
    """
    a, a_single = _batched(system.transition, 3)
    c, c_single = _batched(system.observation, 3)
    k, k_single = _batched(system.gain, 3)
    v, v_single = _batched(system.innovation_covariance, 3)
    system_single = a_single and c_single and k_single and v_single

    matrix = torch.as_tensor(projection, dtype=c.dtype, device=c.device)
    projection_single = matrix.ndim == 2
    if projection_single:
        matrix = matrix.unsqueeze(0)
    if matrix.ndim != 3:
        raise ValueError("projection must have shape (m,n) or (batch,m,n)")
    if matrix.shape[-1] != c.shape[-2]:
        raise ValueError(
            "projection input dimension must match observation dimension"
        )
    if not 1 <= matrix.shape[-2] <= matrix.shape[-1]:
        raise ValueError("projection output dimension must be between 1 and n")
    if not bool(torch.isfinite(matrix).all().item()):
        raise ValueError("projection must contain only finite values")
    if bool(
        torch.any(torch.linalg.matrix_rank(matrix) < matrix.shape[-2]).item()
    ):
        raise ValueError("projection must have full row rank")

    batch = max(
        a.shape[0], c.shape[0], k.shape[0], v.shape[0], matrix.shape[0]
    )
    tensors = (a, c, k, v, matrix)
    if any(x.shape[0] not in (1, batch) for x in tensors):
        raise ValueError("incompatible batch dimensions")
    a, c, k, v, matrix = [
        x.expand(batch, *x.shape[1:]) if x.shape[0] == 1 else x
        for x in tensors
    ]

    projected_c = matrix @ c
    process_q = k @ v @ k.transpose(-1, -2)
    observation_r = symmetrise(
        matrix @ v @ matrix.transpose(-1, -2)
    )
    cross_s = k @ v @ matrix.transpose(-1, -2)

    # The cross covariance S is essential: process and projected observation
    # noise are driven by the same microscopic innovations.
    prediction = solve_generalized_dare(
        a,
        projected_c,
        process_q,
        observation_r,
        cross_s,
    )
    if prediction.ndim == 2:
        prediction = prediction.unsqueeze(0)
    reduced_v = symmetrise(
        projected_c
        @ prediction
        @ projected_c.transpose(-1, -2)
        + observation_r
    )
    gain_numerator = (
        a @ prediction @ projected_c.transpose(-1, -2) + cross_s
    )
    identity = torch.eye(
        reduced_v.shape[-1], dtype=reduced_v.dtype, device=reduced_v.device
    ).expand_as(reduced_v)
    reduced_k = gain_numerator @ spd_solve(reduced_v, identity)

    if system_single and projection_single:
        return InnovationsStateSpace(
            a[0], projected_c[0], reduced_k[0], reduced_v[0]
        )
    return InnovationsStateSpace(a, projected_c, reduced_k, reduced_v)


def reduce_innovations_state_space(
    system: InnovationsStateSpace, indices
) -> InnovationsStateSpace:
    """Obtain an exact marginal innovations model via the projection primitive."""
    index = torch.as_tensor(
        tuple(indices), dtype=torch.long, device=system.observation.device
    )
    n_observations = system.observation.shape[-2]
    if index.ndim != 1 or index.numel() == 0:
        raise ValueError("indices must contain at least one observation index")
    if bool(torch.any((index < 0) | (index >= n_observations)).item()):
        raise ValueError("indices contain an out-of-range observation index")
    selector = torch.eye(
        n_observations,
        dtype=system.observation.dtype,
        device=system.observation.device,
    ).index_select(0, index)
    return _project_innovations_state_space(system, selector)


def innovations_transfer_function(
    system: InnovationsStateSpace, frequencies: torch.Tensor
) -> torch.Tensor:
    """Frequency response H(f)=I+C(zI-A)^-1K for normalized f in [0, .5]."""
    # Evaluate H(z)=I+C(zI-A)^-1 K, the transfer function from innovations to observations.
    a, single = _batched(system.transition, 3)
    c, _ = _batched(system.observation, 3)
    k, _ = _batched(system.gain, 3)
    batch = max(x.shape[0] for x in (a, c, k))
    a, c, k = [
        x.expand(batch, *x.shape[1:]) if x.shape[0] == 1 else x
        for x in (a, c, k)
    ]
    frequencies = torch.as_tensor(
        frequencies, dtype=a.dtype, device=a.device
    )
    complex_dtype = torch.complex128 if a.dtype == torch.float64 else torch.complex64
    a_complex = a.to(complex_dtype)
    c_complex = c.to(complex_dtype)
    k_complex = k.to(complex_dtype)
    state_identity = torch.eye(
        a.shape[-1], dtype=complex_dtype, device=a.device
    )
    observation_identity = torch.eye(
        c.shape[-2], dtype=complex_dtype, device=a.device
    )
    z = torch.exp(2j * torch.pi * frequencies).reshape(1, -1, 1, 1)
    # Solve the linear system directly instead of multiplying by an explicit inverse.
    resolvent = torch.linalg.solve(
        z * state_identity - a_complex[:, None], k_complex[:, None]
    )
    transfer = observation_identity + c_complex[:, None] @ resolvent
    return transfer[0] if single else transfer


def reduce_state_space(system: StateSpaceModel, indices) -> StateSpaceModel:
    """Marginalise observations while preserving latent dynamics."""
    index = torch.as_tensor(
        indices, dtype=torch.long, device=system.observation.device
    )
    observation = system.observation.index_select(-2, index)
    noise = system.observation_covariance.index_select(-2, index).index_select(
        -1, index
    )
    names = (
        None
        if system.channel_names is None
        else tuple(system.channel_names[i] for i in index.tolist())
    )
    return StateSpaceModel(
        system.transition,
        observation,
        system.process_covariance,
        noise,
        system.state_covariance,
        system.sampling_frequency,
        names,
    )


def project_state_space(
    system: StateSpaceModel, projection: torch.Tensor
) -> StateSpaceModel:
    """Apply a linear observation projection while sharing latent dynamics."""
    matrix = torch.as_tensor(
        projection, dtype=system.observation.dtype, device=system.observation.device
    )
    if matrix.ndim == 2 and system.observation.ndim == 3:
        matrix = matrix.unsqueeze(0)
    observation = matrix @ system.observation
    noise = symmetrise(
        matrix
        @ system.observation_covariance
        @ matrix.transpose(-1, -2)
    )
    return StateSpaceModel(
        system.transition,
        observation,
        system.process_covariance,
        noise,
        system.state_covariance,
        system.sampling_frequency,
        None,
    )


def dynamical_dependence(
    system: StateSpaceModel | InnovationsStateSpace | VARSystem,
    projection: torch.Tensor,
    *,
    base: float = 2.0,
) -> torch.Tensor:
    r"""Return linear-Gaussian dynamical dependence for ``Y_t=L X_t``.

    This function returns the Gaussian Granger-causality form of dynamical
    dependence used by SSDI. For a full-row-rank coarse-graining ``Y=LX``,
    Barnett and Seth (2023), Eq. (22), gives

    .. math::

       F(X\to Y)
       = \log\frac{|\Sigma_R|}{|L\Sigma L^\top|},

    where :math:`\Sigma` is the innovations covariance of the microscopic
    process and :math:`\Sigma_R` is the innovations covariance of the reduced
    projected process. Under the standard Gaussian Shannon convention,
    transfer entropy is :math:`T(X\to Y)=F(X\to Y)/2`; the SSDI objective uses
    :math:`F`, so no factor ``1/2`` is applied here.

    Parameters
    ----------
    system
        Microscopic process as a general ``StateSpaceModel``, an
        ``InnovationsStateSpace``, or a canonical ``VARSystem``. General
        state-space models are first converted to innovations form.
    projection
        Required linear coarse-graining ``L`` with shape ``(m,n)`` or
        ``(batch,m,n)``. It must be finite and full row rank. Orthonormal rows
        are not required because DD is invariant under nonsingular changes of
        macro coordinates.
    base
        Logarithm base. The default remains ``2`` for ComplexTorch API
        consistency. Use ``base=np.e`` for the natural-log convention used by
        ComplexBox/SSDI.

    Returns
    -------
    torch.Tensor
        Scalar for an unbatched system/projection or shape ``(batch,)`` after
        broadcasting compatible batch dimensions.

    Notes
    -----
    The reduced model is computed exclusively through
    ``_project_innovations_state_space``. In particular, its generalized DARE
    includes :math:`S=K\Sigma L^\top`, because process and projected
    observation noise share the same microscopic innovations.

    References
    ----------
    - Barnett, L. and Seth, A. K. (2023). *Physical Review E* 108, 014304,
      Eqs. (22), (33), and (34).
    """
    base_value = _validate_log_base(base)
    innovations_system = _as_innovations_state_space(system)
    projected = _project_innovations_state_space(
        innovations_system, projection
    )
    matrix = torch.as_tensor(
        projection,
        dtype=innovations_system.observation.dtype,
        device=innovations_system.observation.device,
    )
    if matrix.ndim == 2:
        matrix = matrix.unsqueeze(0)
    v, v_single = _batched(innovations_system.innovation_covariance, 3)
    batch = max(v.shape[0], matrix.shape[0])
    if v.shape[0] not in (1, batch) or matrix.shape[0] not in (1, batch):
        raise ValueError("incompatible batch dimensions")
    if v.shape[0] == 1:
        v = v.expand(batch, -1, -1)
    if matrix.shape[0] == 1:
        matrix = matrix.expand(batch, -1, -1)
    full_history_v = symmetrise(
        matrix @ v @ matrix.transpose(-1, -2)
    )
    reduced_v = projected.innovation_covariance
    if reduced_v.ndim == 2:
        reduced_v = reduced_v.unsqueeze(0)
    value = (
        spd_logdet(reduced_v) - spd_logdet(full_history_v)
    ) / np.log(base_value)
    return value[0] if v_single and projection.ndim == 2 else value


def stochastic_interaction(
    system: StateSpaceModel, groups, *, base: float = 2.0
) -> torch.Tensor:
    r"""Return Gaussian stochastic interaction from innovation volumes.

    .. math::

       \mathrm{SI}=\sum_g \log|\Sigma_g^R|-\log|\Sigma|,

    where :math:`\Sigma` is the full innovations covariance and each
    :math:`\Sigma_g^R` is the exact reduced innovations covariance for group
    ``g``.
    """
    base_value = _validate_log_base(base)
    full_v = innovations_form(system).covariance
    parts = torch.stack(
        [
            spd_logdet(
                innovations_form(reduce_state_space(system, group)).covariance
            )
            for group in groups
        ],
        -1,
    )
    return (parts.sum(-1) - spd_logdet(full_v)) / np.log(base_value)


@dataclass(frozen=True)
class ProjectionSearchResult:
    """Result of optimizing a macroscopic state-space projection."""

    projection: torch.Tensor
    objective: torch.Tensor
    history: torch.Tensor


def optimise_dynamical_dependence_projection(
    system: StateSpaceModel,
    output_dimension: int,
    *,
    n_candidates: int = 256,
    seed: int = 0,
    minimise: bool = True,
) -> ProjectionSearchResult:
    """Reproducible Stiefel search reusing projection, DARE and DD primitives."""
    # Optimize on the Stiefel manifold so projection columns remain orthonormal throughout the search.
    n_observations = system.observation.shape[-2]
    if not 1 <= output_dimension <= n_observations:
        raise ValueError(
            "output_dimension must be between 1 and observation dimension"
        )
    generator = torch.Generator(device=system.observation.device).manual_seed(seed)
    values, projections = [], []
    for _ in range(n_candidates):
        raw = torch.randn(
            (n_observations, output_dimension),
            generator=generator,
            dtype=system.observation.dtype,
            device=system.observation.device,
        )
        orthogonal, _ = torch.linalg.qr(raw, mode="reduced")
        projection = orthogonal.transpose(-1, -2)
        values.append(dynamical_dependence(system, projection))
        projections.append(projection)
    history = torch.stack(values)
    scores = (
        history.mean(tuple(range(1, history.ndim)))
        if history.ndim > 1
        else history
    )
    index = torch.argmin(scores) if minimise else torch.argmax(scores)
    return ProjectionSearchResult(
        projections[int(index)], history[index], history
    )
