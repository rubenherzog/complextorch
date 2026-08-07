r"""Exact state-space transformations used by multiscale information measures.

The routines in this module reproduce the shared mathematical primitives used
by the Faes ``msID``/``msGC`` MATLAB toolboxes without porting their duplicated
measure pipelines.  They operate on canonical ComplexTorch representations and
reuse the audited generalized-DARE implementation.

References
----------
- Aoki, M. (1990). *State Space Modeling of Time Series*.
- Solo, V. (2016). State-space analysis of Granger-Geweke causality measures
  with application to fMRI. *Neural Computation*, 28, 914-949.
- Faes et al., ``msID`` and ``msGC`` MATLAB toolboxes, ``varma2iss.m`` and
  ``iss_ds.m``.
"""
from __future__ import annotations

from typing import Literal

import torch

from .control import InnovationsStateSpace, solve_generalized_dare
from .linalg import spd_solve, symmetrise


def _normalise_lag_matrices(
    values: torch.Tensor,
    *,
    name: str,
    allow_empty: bool,
) -> tuple[torch.Tensor, bool]:
    """Normalize lag matrices to ``(batch, lag, target, source)``."""
    tensor = torch.as_tensor(values)
    single = tensor.ndim == 3
    if single:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4 or tensor.shape[-1] != tensor.shape[-2]:
        raise ValueError(
            f"{name} must have shape (lags,n,n) or (batch,lags,n,n)"
        )
    if not allow_empty and tensor.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one lag")
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return tensor, single


def _normalise_square_matrix(
    values: torch.Tensor,
    *,
    name: str,
    n_variables: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, bool]:
    """Normalize a square matrix to ``(batch,n,n)`` on a fixed dtype/device."""
    tensor = torch.as_tensor(values, dtype=dtype, device=device)
    single = tensor.ndim == 2
    if single:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or tensor.shape[-2:] != (n_variables, n_variables):
        raise ValueError(
            f"{name} must have shape (n,n) or (batch,n,n) with n={n_variables}"
        )
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return tensor, single


def _broadcast_batch(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Broadcast tensors over one leading batch dimension."""
    batch = max(tensor.shape[0] for tensor in tensors)
    if any(tensor.shape[0] not in (1, batch) for tensor in tensors):
        raise ValueError("incompatible batch dimensions")
    return tuple(
        tensor.expand(batch, *tensor.shape[1:])
        if tensor.shape[0] == 1
        else tensor
        for tensor in tensors
    )


def varma_to_innovations_state_space(
    ar_coefficients: torch.Tensor,
    ma_coefficients: torch.Tensor | None,
    innovation_covariance: torch.Tensor,
    *,
    zero_lag_ma: torch.Tensor | None = None,
) -> InnovationsStateSpace:
    r"""Convert a Gaussian VARMA model to Aoki innovations state-space form.

    The VARMA convention is

    .. math::

       y_t = \sum_{i=1}^{p} A_i y_{t-i}
             + B_0 u_t + \sum_{j=1}^{q} B_j u_{t-j},
       \qquad u_t\sim\mathcal N(0,V).

    ``ar_coefficients`` and ``ma_coefficients`` use ComplexTorch lag-major
    layout ``(..., lag, target, source)``.  The returned innovations process is

    .. math::

       x_{t+1}=A x_t+K\varepsilon_t,\qquad
       y_t=Cx_t+\varepsilon_t,
       \qquad \operatorname{cov}(\varepsilon_t)=R=B_0VB_0^\top.

    This is the Aoki realization used by Faes ``varma2iss.m``.  Unlike the
    MATLAB reference, the MA-state gain block is obtained with a linear solve
    rather than an explicit ``inv(B0)``.

    Parameters
    ----------
    ar_coefficients
        AR matrices with shape ``(p,n,n)`` or ``(batch,p,n,n)``. At least one
        AR lag is currently required, matching the multiscale VAR use case.
    ma_coefficients
        Positive-lag MA matrices ``B_1,...,B_q`` with shape ``(q,n,n)`` or
        ``(batch,q,n,n)``. ``None`` denotes ``q=0``.
    innovation_covariance
        Covariance ``V`` of ``u_t`` with shape ``(n,n)`` or ``(batch,n,n)``.
    zero_lag_ma
        ``B_0`` with shape ``(n,n)`` or ``(batch,n,n)``. Defaults to identity.

    Returns
    -------
    InnovationsStateSpace
        Exact Aoki innovations realization. Entirely unbatched inputs produce
        unbatched matrices; compatible batch dimensions are broadcast.

    Notes
    -----
    The Aoki realization requires nonsingular ``B_0`` when ``q>0`` because its
    MA state is driven by ``B_0^{-1} epsilon_t``.  No explicit matrix inverse is
    formed.
    """
    ar, ar_single = _normalise_lag_matrices(
        ar_coefficients, name="ar_coefficients", allow_empty=False
    )
    if not ar.is_floating_point():
        raise TypeError("VARMA coefficients must use a floating-point dtype")
    batch_ar, order_ar, n_variables, _ = ar.shape

    if ma_coefficients is None:
        ma = torch.empty(
            (1, 0, n_variables, n_variables), dtype=ar.dtype, device=ar.device
        )
        ma_single = True
    else:
        ma, ma_single = _normalise_lag_matrices(
            torch.as_tensor(ma_coefficients, dtype=ar.dtype, device=ar.device),
            name="ma_coefficients",
            allow_empty=True,
        )
        if ma.shape[-2:] != (n_variables, n_variables):
            raise ValueError("AR and MA matrices must have the same variable dimension")

    covariance, covariance_single = _normalise_square_matrix(
        innovation_covariance,
        name="innovation_covariance",
        n_variables=n_variables,
        dtype=ar.dtype,
        device=ar.device,
    )
    if zero_lag_ma is None:
        b0 = torch.eye(n_variables, dtype=ar.dtype, device=ar.device).unsqueeze(0)
        b0_single = True
    else:
        b0, b0_single = _normalise_square_matrix(
            zero_lag_ma,
            name="zero_lag_ma",
            n_variables=n_variables,
            dtype=ar.dtype,
            device=ar.device,
        )

    ar, ma, covariance, b0 = _broadcast_batch(ar, ma, covariance, b0)
    batch = ar.shape[0]
    order_ma = ma.shape[1]
    state_dimension = n_variables * (order_ar + order_ma)

    # MATLAB ``Am=[A1 ... Ap]`` and ``Bm=[B1 ... Bq]`` become the observation
    # row of the Aoki state realization.
    ar_horizontal = ar.permute(0, 2, 1, 3).reshape(
        batch, n_variables, order_ar * n_variables
    )
    if order_ma:
        ma_horizontal = ma.permute(0, 2, 1, 3).reshape(
            batch, n_variables, order_ma * n_variables
        )
        observation = torch.cat((ar_horizontal, ma_horizontal), dim=-1)
    else:
        observation = ar_horizontal

    transition = torch.zeros(
        (batch, state_dimension, state_dimension), dtype=ar.dtype, device=ar.device
    )
    transition[:, :n_variables, : order_ar * n_variables] = ar_horizontal
    if order_ar > 1:
        transition[
            :, n_variables : order_ar * n_variables, : (order_ar - 1) * n_variables
        ] = torch.eye(
            (order_ar - 1) * n_variables, dtype=ar.dtype, device=ar.device
        )

    if order_ma:
        ma_start = order_ar * n_variables
        transition[:, :n_variables, ma_start:] = ma_horizontal
        if order_ma > 1:
            transition[
                :, ma_start + n_variables :, ma_start : ma_start + (order_ma - 1) * n_variables
            ] = torch.eye(
                (order_ma - 1) * n_variables, dtype=ar.dtype, device=ar.device
            )

    innovation = symmetrise(b0 @ covariance @ b0.transpose(-1, -2))
    gain = torch.zeros(
        (batch, state_dimension, n_variables), dtype=ar.dtype, device=ar.device
    )
    identity = torch.eye(n_variables, dtype=ar.dtype, device=ar.device).expand(
        batch, n_variables, n_variables
    )
    gain[:, :n_variables, :] = identity
    if order_ma:
        try:
            # Exact counterpart of MATLAB ``inv(B0)`` without forming an inverse.
            b0_solve = torch.linalg.solve(b0, identity)
        except RuntimeError as exc:
            raise ValueError(
                "zero_lag_ma must be nonsingular for an MA realization"
            ) from exc
        gain[
            :, order_ar * n_variables : (order_ar + 1) * n_variables, :
        ] = b0_solve

    single = ar_single and ma_single and covariance_single and b0_single
    if single:
        return InnovationsStateSpace(
            transition[0], observation[0], gain[0], innovation[0]
        )
    return InnovationsStateSpace(transition, observation, gain, innovation)


def downsample_innovations_state_space(
    system: InnovationsStateSpace,
    factor: int,
    *,
    backend: Literal["scipy", "torch"] = "torch",
    rtol: float | None = None,
    atol: float | None = None,
    max_iter: int = 100,
) -> InnovationsStateSpace:
    r"""Return the exact innovations model sampled every ``factor`` steps.

    For

    .. math::

       x_{t+1}=Ax_t+K\varepsilon_t,\qquad
       y_t=Cx_t+\varepsilon_t,\qquad
       \operatorname{cov}(\varepsilon_t)=V,

    sampling every :math:`k` observations gives the general state-space
    parameters used by Solo and by Faes ``iss_ds.m``:

    .. math::

       A_k &= A^k,\\
       Q_k &= \sum_{j=0}^{k-1} A^j KVK^\top(A^j)^\top,\\
       S_k &= A^{k-1}KV,\\
       C_k &= C,\qquad R_k=V.

    The exact innovations gain and covariance are then recovered with the
    canonical ComplexTorch generalized DARE.

    Parameters
    ----------
    system
        Innovations-form state-space process, batched or unbatched.
    factor
        Positive integer downsampling factor. ``factor=1`` returns ``system``.
    backend
        Generalized-DARE backend, ``"torch"`` or SciPy reference ``"scipy"``.
    rtol, atol, max_iter
        Numerical controls forwarded to ``solve_generalized_dare``.

    Returns
    -------
    InnovationsStateSpace
        Exact downsampled innovations process with the same observation matrix.

    References
    ----------
    - Solo (2016), Sec. 4.
    - Faes ``msID``/``msGC`` ``iss_ds.m``.
    """
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
        raise ValueError("factor must be a positive integer")
    if factor == 1:
        return system

    a = torch.as_tensor(system.transition)
    c = torch.as_tensor(system.observation)
    k_gain = torch.as_tensor(system.gain)
    v = torch.as_tensor(system.innovation_covariance)
    single_flags = (a.ndim == 2, c.ndim == 2, k_gain.ndim == 2, v.ndim == 2)
    if single_flags[0]:
        a = a.unsqueeze(0)
    if single_flags[1]:
        c = c.unsqueeze(0)
    if single_flags[2]:
        k_gain = k_gain.unsqueeze(0)
    if single_flags[3]:
        v = v.unsqueeze(0)
    if a.ndim != 3 or a.shape[-1] != a.shape[-2]:
        raise ValueError("transition must have shape (r,r) or (batch,r,r)")
    if c.ndim != 3 or c.shape[-1] != a.shape[-1]:
        raise ValueError("observation must have shape (n,r) or (batch,n,r)")
    if k_gain.ndim != 3 or k_gain.shape[-2:] != (a.shape[-1], c.shape[-2]):
        raise ValueError("gain must have shape (r,n) or (batch,r,n)")
    if v.ndim != 3 or v.shape[-2:] != (c.shape[-2], c.shape[-2]):
        raise ValueError(
            "innovation_covariance must have shape (n,n) or (batch,n,n)"
        )
    if not all(tensor.is_floating_point() for tensor in (a, c, k_gain, v)):
        raise TypeError("innovations state-space matrices must be floating point")
    if len({tensor.dtype for tensor in (a, c, k_gain, v)}) != 1:
        raise ValueError("innovations state-space matrices must use the same dtype")
    if len({tensor.device for tensor in (a, c, k_gain, v)}) != 1:
        raise ValueError("innovations state-space matrices must use the same device")
    if not all(bool(torch.isfinite(tensor).all().item()) for tensor in (a, c, k_gain, v)):
        raise ValueError("innovations state-space matrices must contain only finite values")

    a, c, k_gain, v = _broadcast_batch(a, c, k_gain, v)
    batch, state_dimension, _ = a.shape
    identity = torch.eye(
        state_dimension, dtype=a.dtype, device=a.device
    ).expand(batch, state_dimension, state_dimension)

    # Build the exact aggregated process covariance without the Cholesky-only
    # construction used by MATLAB: Q_k=sum_j A^j K V K' (A^j)'.
    process_covariance = symmetrise(k_gain @ v @ k_gain.transpose(-1, -2))
    q_down = process_covariance.clone()
    a_power = identity
    for _ in range(1, factor):
        a_power = a_power @ a
        q_down = q_down + a_power @ process_covariance @ a_power.transpose(-1, -2)
    q_down = symmetrise(q_down)

    # At loop exit A_power=A^(k-1), matching Faes ``Sk = Ak*K*V`` before
    # ``Ak = Ak*A`` in iss_ds.m.
    cross_covariance = a_power @ k_gain @ v
    transition_down = a_power @ a

    prediction_covariance = solve_generalized_dare(
        transition_down,
        c,
        q_down,
        v,
        cross_covariance,
        backend=backend,
        rtol=rtol,
        atol=atol,
        max_iter=max_iter,
    )
    if prediction_covariance.ndim == 2:
        prediction_covariance = prediction_covariance.unsqueeze(0)
    innovation_down = symmetrise(
        c @ prediction_covariance @ c.transpose(-1, -2) + v
    )
    numerator = (
        transition_down @ prediction_covariance @ c.transpose(-1, -2)
        + cross_covariance
    )
    gain_down = spd_solve(
        innovation_down, numerator.transpose(-1, -2)
    ).transpose(-1, -2)

    single = all(single_flags)
    if single:
        return InnovationsStateSpace(
            transition_down[0], c[0], gain_down[0], innovation_down[0]
        )
    return InnovationsStateSpace(
        transition_down, c, gain_down, innovation_down
    )
