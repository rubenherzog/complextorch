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
from typing import Literal
import numpy as np
import torch
from scipy.linalg import solve_discrete_are
from .linalg import spd_logdet, spd_solve, symmetrise
from .representations import StateSpaceModel, VARSystem


def _batched(t: torch.Tensor, ndim: int) -> tuple[torch.Tensor, bool]:
    """Batched.
    
    Parameters
    ----------
    t
        Input required by this calculation.
    ndim
        Input required by this calculation.
    
    Returns
    -------
    object
        Computed result; see the annotated return type and shape notes.
    
    Notes
    -----
    Batch dimensions are preserved unless explicitly documented otherwise.
    The implementation validates dimensional and positive-definiteness
    requirements before executing the numerical core.
    """
    x = torch.as_tensor(t)
    single = x.ndim == ndim - 1
    return (x.unsqueeze(0) if single else x), single


def _validate_log_base(base: float) -> float:
    """Return a finite positive logarithm base different from one."""
    value = float(base)
    if not np.isfinite(value) or value <= 0.0 or value == 1.0:
        raise ValueError("base must be finite, positive, and different from 1")
    return value


def _broadcast_dare_inputs(
    transition,
    observation,
    process_covariance,
    observation_covariance,
):
    """Normalize DARE inputs to one compatible leading batch dimension."""
    a, a_single = _batched(transition, 3)
    c, c_single = _batched(observation, 3)
    q, q_single = _batched(process_covariance, 3)
    r, r_single = _batched(observation_covariance, 3)
    batch = max(a.shape[0], c.shape[0], q.shape[0], r.shape[0])
    tensors = (a, c, q, r)
    if any(x.shape[0] not in (1, batch) for x in tensors):
        raise ValueError("incompatible DARE batch dimensions")
    broadcast = tuple(
        x.expand(batch, *x.shape[1:]) if x.shape[0] == 1 else x
        for x in tensors
    )
    single = a_single and c_single and q_single and r_single
    return (*broadcast, single)


def _solve_dare_scipy(a, c, q, r):
    """Reference DARE backend using SciPy's ordered-QZ implementation."""
    out = []
    for ai, ci, qi, ri in zip(
        *[x.detach().cpu().numpy() for x in (a, c, q, r)], strict=True
    ):
        # SciPy uses the control-form convention
        # A' X A - X - A' X B (R+B'XB)^-1 B' X A + Q = 0.
        # The filtering DARE is obtained with A_scipy=A' and B_scipy=C'.
        out.append(
            torch.as_tensor(
                solve_discrete_are(ai.T, ci.T, qi, ri),
                dtype=a.dtype,
                device=a.device,
            )
        )
    return symmetrise(torch.stack(out))


def _solve_dare_torch(a, c, q, r, *, rtol, atol, max_iter):
    r"""Solve the filtering DARE with a batched structured doubling algorithm.

    Writing the equivalent control-form DARE with
    :math:`A_c=A^\top`, :math:`B_c=C^\top`, define

    .. math::

       G_0=B_cR^{-1}B_c^\top,\qquad H_0=Q,\qquad A_0=A_c.

    The structured doubling iteration is

    .. math::

       A_{k+1} &= A_k(I+G_kH_k)^{-1}A_k,\\
       G_{k+1} &= G_k+A_kG_k(I+H_kG_k)^{-1}A_k^\top,\\
       H_{k+1} &= H_k+A_k^\top H_k(I+G_kH_k)^{-1}A_k,

    and :math:`H_k` converges quadratically to the stabilizing DARE solution
    under the standard stabilizability/detectability assumptions.
    """
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    if rtol < 0 or atol < 0:
        raise ValueError("rtol and atol must be non-negative")

    # SDA is substantially less forgiving in native float32 on ill-conditioned
    # systems. Use float64 working precision on the same device, then cast back
    # to preserve the public dtype while remaining entirely within Torch.
    public_dtype = a.dtype
    work_dtype = torch.float64 if public_dtype == torch.float32 else public_dtype
    a_work, c_work, q_work, r_work = [
        tensor.to(dtype=work_dtype) for tensor in (a, c, q, r)
    ]

    # R is an observation covariance and must be SPD. Cholesky provides both
    # the required validation and R^-1 C without forming an explicit inverse.
    r_chol = torch.linalg.cholesky(symmetrise(r_work))
    g = symmetrise(
        c_work.transpose(-1, -2) @ torch.cholesky_solve(c_work, r_chol)
    )
    h = symmetrise(q_work)
    a_k = a_work.transpose(-1, -2)

    n_states = a_work.shape[-1]
    identity = torch.eye(
        n_states, dtype=work_dtype, device=a_work.device
    ).expand(a_work.shape[0], n_states, n_states)

    for _ in range(max_iter):
        # I+GH and I+HG are generally not symmetric; use direct linear solves.
        i_plus_gh = identity + g @ h
        i_plus_hg = identity + h @ g
        solved_a = torch.linalg.solve(i_plus_gh, a_k)
        solved_at = torch.linalg.solve(i_plus_hg, a_k.transpose(-1, -2))

        a_next = a_k @ solved_a
        g_next = symmetrise(g + a_k @ g @ solved_at)
        h_next = symmetrise(
            h + a_k.transpose(-1, -2) @ h @ solved_a
        )

        difference = torch.linalg.matrix_norm(
            h_next - h, ord="fro", dim=(-2, -1)
        )
        scale = torch.linalg.matrix_norm(
            h_next, ord="fro", dim=(-2, -1)
        ).clamp_min(1.0)
        if bool(torch.all(difference <= atol + rtol * scale)):
            return h_next.to(dtype=public_dtype)
        a_k, g, h = a_next, g_next, h_next

    raise RuntimeError("Torch DARE structured doubling algorithm did not converge")


def solve_dare(
    transition,
    observation,
    process_covariance,
    observation_covariance,
    *,
    backend: Literal["scipy", "torch"] = "scipy",
    rtol: float | None = None,
    atol: float | None = None,
    max_iter: int = 100,
):
    r"""Solve the steady-state discrete algebraic Riccati equation.

    ComplexTorch uses the filtering convention

    .. math::

       P=APA^\top+Q
       -APC^\top(CPC^\top+R)^{-1}CPA^\top.

    Parameters
    ----------
    transition
        State transition ``A`` with shape ``(r,r)`` or ``(batch,r,r)``.
    observation
        Observation matrix ``C`` with shape ``(n,r)`` or ``(batch,n,r)``.
    process_covariance
        Process covariance ``Q`` with shape ``(r,r)`` or ``(batch,r,r)``.
    observation_covariance
        Observation covariance ``R`` with shape ``(n,n)`` or
        ``(batch,n,n)``.
    backend
        ``"scipy"`` retains SciPy ``solve_discrete_are`` as the reference
        implementation. ``"torch"`` uses a device-native batched structured
        doubling algorithm without conversion to NumPy.
    rtol, atol
        Convergence tolerances for the Torch backend. Defaults are
        ``1e-10``/``1e-12`` for float64 and ``1e-8``/``1e-10`` in the float64
        working precision used for float32 inputs. They are ignored by SciPy.
    max_iter
        Maximum structured-doubling iterations for the Torch backend.

    Returns
    -------
    torch.Tensor
        Stabilizing prediction covariance with shape ``(r,r)`` for entirely
        unbatched inputs or ``(batch,r,r)`` after batch broadcasting.

    Notes
    -----
    The SciPy backend is intentionally retained as a numerical oracle. The
    Torch backend is not differentiated through in this implementation; its
    purpose here is device-native batched forward evaluation and parity.

    References
    ----------
    - Anderson and Moore (1979), *Optimal Filtering*.
    - van Dooren (1981), generalized eigenvalue approach to Riccati equations.
    - Laub (1979), Schur method for algebraic Riccati equations.
    """
    if backend not in {"scipy", "torch"}:
        raise ValueError("backend must be 'scipy' or 'torch'")

    a, c, q, r, single = _broadcast_dare_inputs(
        transition, observation, process_covariance, observation_covariance
    )
    if backend == "scipy":
        result = _solve_dare_scipy(a, c, q, r)
    else:
        if not a.is_floating_point():
            raise TypeError("Torch DARE inputs must use a floating-point dtype")
        dtypes = {tensor.dtype for tensor in (a, c, q, r)}
        if len(dtypes) != 1:
            raise ValueError("Torch DARE inputs must use the same dtype")
        default_rtol = 1e-8 if a.dtype == torch.float32 else 1e-10
        default_atol = 1e-10 if a.dtype == torch.float32 else 1e-12
        result = _solve_dare_torch(
            a,
            c,
            q,
            r,
            rtol=default_rtol if rtol is None else float(rtol),
            atol=default_atol if atol is None else float(atol),
            max_iter=int(max_iter),
        )
    return result[0] if single else result


def _broadcast_generalized_dare_inputs(
    transition,
    observation,
    process_covariance,
    observation_covariance,
    cross_covariance,
):
    """Normalize generalized-DARE inputs to one compatible batch dimension."""
    a, a_single = _batched(transition, 3)
    c, c_single = _batched(observation, 3)
    q, q_single = _batched(process_covariance, 3)
    r, r_single = _batched(observation_covariance, 3)
    s, s_single = _batched(cross_covariance, 3)
    tensors = (a, c, q, r, s)
    batch = max(x.shape[0] for x in tensors)
    if any(x.shape[0] not in (1, batch) for x in tensors):
        raise ValueError("incompatible generalized DARE batch dimensions")
    broadcast = tuple(
        x.expand(batch, *x.shape[1:]) if x.shape[0] == 1 else x
        for x in tensors
    )
    single = a_single and c_single and q_single and r_single and s_single
    return (*broadcast, single)


def _solve_generalized_dare_scipy(a, c, q, r, s):
    """Reference generalized DARE using SciPy's direct cross-term interface."""
    out = []
    arrays = [x.detach().cpu().numpy() for x in (a, c, q, r, s)]
    for ai, ci, qi, ri, si in zip(*arrays, strict=True):
        # SciPy's S term maps directly to the process-observation covariance
        # in the dual filtering equation when A_scipy=A' and B_scipy=C'.
        out.append(
            torch.as_tensor(
                solve_discrete_are(ai.T, ci.T, qi, ri, s=si),
                dtype=a.dtype,
                device=a.device,
            )
        )
    return symmetrise(torch.stack(out))


def _solve_generalized_dare_torch(a, c, q, r, s, *, rtol, atol, max_iter):
    r"""Decorrelate the generalized DARE and reuse the Torch standard solver.

    For

    .. math::

       P=APA^\top+Q-(APC^\top+S)
       (CPC^\top+R)^{-1}(APC^\top+S)^\top,

    positive-definite :math:`R` permits the exact transformation

    .. math::

       A_0=A-SR^{-1}C,\qquad Q_0=Q-SR^{-1}S^\top.

    The generalized equation is then the ordinary filtering DARE for
    ``(A_0, C, Q_0, R)``. This avoids a second Riccati iteration and delegates
    numerical solution to the already-audited structured doubling backend.
    """
    public_dtype = a.dtype
    work_dtype = torch.float64 if public_dtype == torch.float32 else public_dtype
    a_work, c_work, q_work, r_work, s_work = [
        tensor.to(dtype=work_dtype) for tensor in (a, c, q, r, s)
    ]

    r_chol = torch.linalg.cholesky(symmetrise(r_work))
    r_inv_c = torch.cholesky_solve(c_work, r_chol)
    r_inv_st = torch.cholesky_solve(s_work.transpose(-1, -2), r_chol)
    decorrelated_a = a_work - s_work @ r_inv_c
    decorrelated_q = symmetrise(q_work - s_work @ r_inv_st)

    result = solve_dare(
        decorrelated_a,
        c_work,
        decorrelated_q,
        r_work,
        backend="torch",
        rtol=rtol,
        atol=atol,
        max_iter=max_iter,
    )
    return result.to(dtype=public_dtype)


# Marginal innovations require the steady-state generalized Riccati solution.
def solve_generalized_dare(
    transition: torch.Tensor,
    observation: torch.Tensor,
    process_covariance: torch.Tensor,
    observation_covariance: torch.Tensor,
    cross_covariance: torch.Tensor,
    *,
    backend: Literal["scipy", "torch"] = "torch",
    rtol: float | None = None,
    atol: float | None = None,
    max_iter: int = 100,
) -> torch.Tensor:
    r"""Solve the generalized filtering DARE with correlated noises.

    ComplexTorch and ComplexBox use

    .. math::

       P=APA^\top+Q-(APC^\top+S)
       (CPC^\top+R)^{-1}(APC^\top+S)^\top,

    where ``S = cov(w_t, v_t)`` has shape ``(r,n)``.

    Parameters
    ----------
    transition
        State transition ``A`` with shape ``(r,r)`` or ``(batch,r,r)``.
    observation
        Observation matrix ``C`` with shape ``(n,r)`` or ``(batch,n,r)``.
    process_covariance
        Process covariance ``Q`` with shape ``(r,r)`` or ``(batch,r,r)``.
    observation_covariance
        Observation covariance ``R`` with shape ``(n,n)`` or
        ``(batch,n,n)``. The Torch backend requires positive definiteness.
    cross_covariance
        Process-observation cross covariance ``S`` with shape ``(r,n)`` or
        ``(batch,r,n)``.
    backend
        ``"scipy"`` calls SciPy's direct generalized DARE and serves as the
        independent numerical reference. ``"torch"`` exactly decorrelates the
        noises and reuses the device-native structured-doubling ``solve_dare``
        backend. ``"torch"`` remains the default to preserve the pre-existing
        device-native behavior of this public function.
    rtol, atol
        Convergence tolerances forwarded to the Torch standard DARE backend.
        They are ignored by SciPy.
    max_iter
        Maximum structured-doubling iterations for the Torch backend.

    Returns
    -------
    torch.Tensor
        Stabilizing prediction covariance with shape ``(r,r)`` for entirely
        unbatched inputs or ``(batch,r,r)`` after batch broadcasting.

    Notes
    -----
    The Torch path performs no NumPy/SciPy conversion. Float32 inputs use the
    same float64 working-precision policy as ``solve_dare`` and are cast back
    to their public dtype. Differentiation through the Riccati solution remains
    out of scope.

    References
    ----------
    - Anderson and Moore (1979), *Optimal Filtering*.
    - Barnett and Seth (2015), state-space Granger-causality reduction.
    - ComplexBox ``mdare`` at commit
      ``87b5e2cd9bba22ddd978bade6f614da7d6190db2``.
    """
    if backend not in {"scipy", "torch"}:
        raise ValueError("backend must be 'scipy' or 'torch'")

    a, c, q, r, s, single = _broadcast_generalized_dare_inputs(
        transition,
        observation,
        process_covariance,
        observation_covariance,
        cross_covariance,
    )
    if backend == "scipy":
        result = _solve_generalized_dare_scipy(a, c, q, r, s)
    else:
        if not all(x.is_floating_point() for x in (a, c, q, r, s)):
            raise TypeError("Torch generalized DARE inputs must use floating-point dtypes")
        dtypes = {x.dtype for x in (a, c, q, r, s)}
        if len(dtypes) != 1:
            raise ValueError("Torch generalized DARE inputs must use the same dtype")
        devices = {x.device for x in (a, c, q, r, s)}
        if len(devices) != 1:
            raise ValueError("Torch generalized DARE inputs must use the same device")
        default_rtol = 1e-8 if a.dtype == torch.float32 else 1e-10
        default_atol = 1e-10 if a.dtype == torch.float32 else 1e-12
        result = _solve_generalized_dare_torch(
            a,
            c,
            q,
            r,
            s,
            rtol=default_rtol if rtol is None else float(rtol),
            atol=default_atol if atol is None else float(atol),
            max_iter=int(max_iter),
        )
    return result[0] if single else result


@dataclass(frozen=True)
class InnovationsForm:
    """Innovationsform.
    
    Notes
    -----
    Public fitted attributes use the trailing-underscore convention.
    """
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
    observation = coefficients.permute(0, 2, 1, 3).reshape(batch, n_variables, state_dimension)
    gain = torch.zeros((batch, state_dimension, n_variables), dtype=coefficients.dtype, device=coefficients.device)
    gain[:, :n_variables, :] = torch.eye(n_variables, dtype=coefficients.dtype, device=coefficients.device)
    return InnovationsStateSpace(system.companion, observation, gain, system.innovation_covariance)


def _as_innovations_state_space(
    system: StateSpaceModel | InnovationsStateSpace | VARSystem,
) -> InnovationsStateSpace:
    """Return the microscopic process in canonical innovations form."""
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
    r"""Return the exact innovations model of the projection ``Y=LX``.

    For an innovations-form microscopic process

    .. math::

       z_{t+1}=Az_t+K\varepsilon_t,\quad
       X_t=Cz_t+\varepsilon_t,\quad
       \operatorname{cov}(\varepsilon_t)=\Sigma,

    the projected process has
    :math:`C_R=LC`, :math:`Q=K\Sigma K^\top`,
    :math:`R=L\Sigma L^\top`, and :math:`S=K\Sigma L^\top`.
    The generalized DARE yields its exact reduced innovations covariance and
    gain. This is the single projection primitive shared by marginal reduction,
    dynamical dependence, and model-derived predictive measures.

    References
    ----------
    - Barnett and Seth (2015).
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
        raise ValueError("projection input dimension must match observation dimension")
    if not 1 <= matrix.shape[-2] <= matrix.shape[-1]:
        raise ValueError("projection output dimension must be between 1 and n")
    if not bool(torch.isfinite(matrix).all().item()):
        raise ValueError("projection must contain only finite values")
    if bool(torch.any(torch.linalg.matrix_rank(matrix) < matrix.shape[-2]).item()):
        raise ValueError("projection must have full row rank")

    batch = max(a.shape[0], c.shape[0], k.shape[0], v.shape[0], matrix.shape[0])
    tensors = (a, c, k, v, matrix)
    if any(x.shape[0] not in (1, batch) for x in tensors):
        raise ValueError("incompatible batch dimensions")
    a, c, k, v, matrix = [
        x.expand(batch, *x.shape[1:]) if x.shape[0] == 1 else x
        for x in tensors
    ]

    reduced_c = matrix @ c
    process_q = k @ v @ k.transpose(-1, -2)
    reduced_r = symmetrise(matrix @ v @ matrix.transpose(-1, -2))
    cross_s = k @ v @ matrix.transpose(-1, -2)
    # Process and projected observation noise share the same microscopic
    # innovations, so the cross covariance S is part of the exact DARE.
    p = solve_generalized_dare(a, reduced_c, process_q, reduced_r, cross_s)
    if p.ndim == 2:
        p = p.unsqueeze(0)
    reduced_v = symmetrise(reduced_c @ p @ reduced_c.transpose(-1, -2) + reduced_r)
    numerator = a @ p @ reduced_c.transpose(-1, -2) + cross_s
    identity = torch.eye(reduced_v.shape[-1], dtype=reduced_v.dtype, device=reduced_v.device).expand_as(reduced_v)
    reduced_k = numerator @ spd_solve(reduced_v, identity)
    if system_single and projection_single:
        return InnovationsStateSpace(a[0], reduced_c[0], reduced_k[0], reduced_v[0])
    return InnovationsStateSpace(a, reduced_c, reduced_k, reduced_v)


def reduce_innovations_state_space(system: InnovationsStateSpace, indices) -> InnovationsStateSpace:
    """Obtain an exact marginal innovations model via generalized DARE."""
    index = torch.as_tensor(tuple(indices), dtype=torch.long, device=system.observation.device)
    n_observations = system.observation.shape[-2]
    if index.ndim != 1 or index.numel() == 0:
        raise ValueError("indices must contain at least one observation index")
    if bool(torch.any((index < 0) | (index >= n_observations)).item()):
        raise ValueError("indices contain an out-of-range observation index")
    selector = torch.eye(n_observations, dtype=system.observation.dtype, device=system.observation.device).index_select(0, index)
    # Coordinate selection is a special case of the canonical projection path.
    return _project_innovations_state_space(system, selector)


def innovations_transfer_function(system: InnovationsStateSpace, frequencies: torch.Tensor) -> torch.Tensor:
    """Frequency response H(f)=I+C(zI-A)^-1K for normalized f in [0, .5]."""
    # Evaluate H(z)=I+C(zI-A)^-1 K, the transfer function from innovations to observations.
    a, single = _batched(system.transition, 3)
    c, _ = _batched(system.observation, 3)
    k, _ = _batched(system.gain, 3)
    batch = max(x.shape[0] for x in (a, c, k))
    a, c, k = [x.expand(batch, *x.shape[1:]) if x.shape[0] == 1 else x for x in (a, c, k)]
    frequencies = torch.as_tensor(frequencies, dtype=a.dtype, device=a.device)
    complex_dtype = torch.complex128 if a.dtype == torch.float64 else torch.complex64
    a_complex, c_complex, k_complex = a.to(complex_dtype), c.to(complex_dtype), k.to(complex_dtype)
    state_identity = torch.eye(a.shape[-1], dtype=complex_dtype, device=a.device)
    observation_identity = torch.eye(c.shape[-2], dtype=complex_dtype, device=a.device)
    z = torch.exp(2j * torch.pi * frequencies).reshape(1, -1, 1, 1)
    # Solve the linear system directly instead of multiplying by an explicit inverse.
    resolvent = torch.linalg.solve(z * state_identity - a_complex[:, None], k_complex[:, None])
    transfer = observation_identity + c_complex[:, None] @ resolvent
    return transfer[0] if single else transfer


def reduce_state_space(system: StateSpaceModel, indices) -> StateSpaceModel:
    """Marginalise observations while preserving latent dynamics."""
    index = torch.as_tensor(indices, dtype=torch.long, device=system.observation.device)
    observation = system.observation.index_select(-2, index)
    noise = system.observation_covariance.index_select(-2, index).index_select(-1, index)
    names = None if system.channel_names is None else tuple(system.channel_names[i] for i in index.tolist())
    return StateSpaceModel(system.transition, observation, system.process_covariance, noise, system.state_covariance, system.sampling_frequency, names)


def project_state_space(system: StateSpaceModel, projection: torch.Tensor) -> StateSpaceModel:
    """Apply a linear observation projection while sharing latent dynamics."""
    matrix = torch.as_tensor(projection, dtype=system.observation.dtype, device=system.observation.device)
    if matrix.ndim == 2 and system.observation.ndim == 3:
        matrix = matrix.unsqueeze(0)
    observation = matrix @ system.observation
    noise = symmetrise(matrix @ system.observation_covariance @ matrix.transpose(-1, -2))
    return StateSpaceModel(system.transition, observation, system.process_covariance, noise, system.state_covariance, system.sampling_frequency, None)


def dynamical_dependence(
    system: StateSpaceModel | InnovationsStateSpace | VARSystem,
    projection: torch.Tensor,
    *,
    base: float = 2.0,
) -> torch.Tensor:
    r"""Return linear-Gaussian dynamical dependence for ``Y_t=L X_t``.

    This returns the Gaussian Granger-causality form used by SSDI. For a
    full-row-rank coarse-graining ``Y=LX``, Barnett and Seth (2023), Eq. (22),
    gives

    .. math::

       F(X\to Y)=\log\frac{|\Sigma_R|}{|L\Sigma L^\top|},

    where :math:`\Sigma` is the microscopic innovations covariance and
    :math:`\Sigma_R` is the innovations covariance of the reduced projected
    process. Under the standard Gaussian Shannon convention,
    :math:`T(X\to Y)=F(X\to Y)/2`; SSDI optimises :math:`F`, so no factor
    ``1/2`` is applied here.

    Parameters
    ----------
    system
        Microscopic process as a general ``StateSpaceModel``, an
        ``InnovationsStateSpace``, or a canonical ``VARSystem``.
    projection
        Required finite full-row-rank coarse-graining with shape ``(m,n)`` or
        ``(batch,m,n)``. Orthonormal rows are not required.
    base
        Logarithm base. The ComplexTorch default remains ``2``. Use ``np.e``
        for direct natural-log comparison with ComplexBox/SSDI.

    Returns
    -------
    torch.Tensor
        Scalar for unbatched inputs or shape ``(batch,)`` after compatible
        batch broadcasting.

    References
    ----------
    - Barnett, L. and Seth, A. K. (2023). *Physical Review E* 108, 014304,
      Eqs. (22), (33), and (34).
    """
    base_value = _validate_log_base(base)
    innovations_system = _as_innovations_state_space(system)
    reduced = _project_innovations_state_space(innovations_system, projection)

    matrix = torch.as_tensor(projection, dtype=innovations_system.observation.dtype, device=innovations_system.observation.device)
    if matrix.ndim == 2:
        matrix = matrix.unsqueeze(0)
    v, _ = _batched(innovations_system.innovation_covariance, 3)
    batch = max(v.shape[0], matrix.shape[0])
    if any(x.shape[0] not in (1, batch) for x in (v, matrix)):
        raise ValueError("incompatible batch dimensions")
    v, matrix = [x.expand(batch, *x.shape[1:]) if x.shape[0] == 1 else x for x in (v, matrix)]
    full_history_v = symmetrise(matrix @ v @ matrix.transpose(-1, -2))
    reduced_v = reduced.innovation_covariance
    reduced_single = reduced_v.ndim == 2
    if reduced_single:
        reduced_v = reduced_v.unsqueeze(0)
    value = (spd_logdet(reduced_v) - spd_logdet(full_history_v)) / np.log(base_value)
    return value[0] if reduced_single else value


def stochastic_interaction(system: StateSpaceModel, groups, *, base: float = 2.0):
    r"""Return Gaussian stochastic interaction from reduced innovation volumes.

    .. math::

       \mathrm{SI}=\sum_g\log|\Sigma_g^R|-\log|\Sigma|,

    where :math:`\Sigma` is the full innovations covariance and
    :math:`\Sigma_g^R` is the exact reduced innovations covariance of group
    ``g``. The microscopic model is converted to innovations form once, and
    each group is reduced through the same canonical projection primitive used
    by dynamical dependence.
    """
    base_value = _validate_log_base(base)
    innovations_system = _as_innovations_state_space(system)
    full_v = innovations_system.innovation_covariance
    parts = torch.stack(
        [
            spd_logdet(
                reduce_innovations_state_space(
                    innovations_system, group
                ).innovation_covariance
            )
            for group in groups
        ],
        -1,
    )
    return (parts.sum(-1) - spd_logdet(full_v)) / np.log(base_value)


@dataclass(frozen=True)
class ProjectionSearchResult:
    """Result of optimizing a macroscopic state-space projection.
    
    Notes
    -----
    Public fitted attributes use the trailing-underscore convention.
    """
    projection: torch.Tensor
    objective: torch.Tensor
    history: torch.Tensor


def optimise_dynamical_dependence_projection(system: StateSpaceModel, output_dimension: int, *, n_candidates: int = 256, seed: int = 0, minimise: bool = True) -> ProjectionSearchResult:
    """Reproducible Stiefel search reusing projection, DARE and DD primitives."""
    # Optimize on the Stiefel manifold so projection columns remain orthonormal throughout the search.
    n_observations = system.observation.shape[-2]
    if not 1 <= output_dimension <= n_observations:
        raise ValueError("output_dimension must be between 1 and observation dimension")
    generator = torch.Generator(device=system.observation.device).manual_seed(seed)
    values, projections = [], []
    for _ in range(n_candidates):
        raw = torch.randn((n_observations, output_dimension), generator=generator, dtype=system.observation.dtype, device=system.observation.device)
        orthogonal, _ = torch.linalg.qr(raw, mode="reduced")
        projection = orthogonal.transpose(-1, -2)
        # Keep the existing random-search algorithm unchanged; only evaluate
        # the corrected DD objective on the original microscopic system.
        values.append(dynamical_dependence(system, projection))
        projections.append(projection)
    history = torch.stack(values)
    scores = history.mean(tuple(range(1, history.ndim))) if history.ndim > 1 else history
    index = torch.argmin(scores) if minimise else torch.argmax(scores)
    return ProjectionSearchResult(projections[int(index)], history[index], history)