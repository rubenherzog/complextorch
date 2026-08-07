"""Grassmannian optimisation primitives for dynamical dependence.

This module ports the SSDI/ComplexBox gradient-descent objectives and update
rules to native Torch tensors while preserving ComplexTorch's row-projection
convention. ComplexBox represents a subspace by columns ``L`` with shape
``(n, m)``; ComplexTorch represents the same map by rows ``M=L.T`` with shape
``(m, n)``.

ComplexBox's proxy and spectral objectives are written for identity innovations.
For a general innovations covariance ``V > 0`` this module applies an exact
observation-space whitening transformation before optimization and maps the
optimized Grassmann subspace back to the original observation coordinates.
No approximation to the objective or dynamics is introduced by this change.

References
----------
- Barnett, L. and Seth, A. K. (2023). Dynamical independence: Discovering
  emergent macroscopic processes in complex dynamical systems.
- bmilinkovic/complexbox, ``src/complexbox/ssdi/dd.py``, ``optimise.py`` and
  ``_torch.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .control import (
    InnovationsStateSpace,
    _as_innovations_state_space,
    innovations_transfer_function,
)
from .representations import StateSpaceModel, VARSystem

Model = StateSpaceModel | InnovationsStateSpace | VARSystem


@dataclass(frozen=True)
class DDGradientSearchResult:
    """Result of a batched SSDI gradient-descent search.

    Attributes
    ----------
    objective
        Sorted final objective values, shape ``(runs,)``.
    projection
        Corresponding row projections in the original observation coordinates,
        shape ``(runs, m, n)``.
    convergence
        SSDI convergence codes: 0 unconverged, 1 step size below tolerance,
        2 objective below tolerance, 3 gradient norm below tolerance.
    step_size
        Final step size for each sorted run.
    iterations
        Iteration at which each run stopped.
    history
        Optional tensor ``(runs, max_recorded_iterations, 3)`` containing
        ``(objective, step_size, gradient_norm)``. Entries after a run stops
        remain ``nan``.
    """

    objective: torch.Tensor
    projection: torch.Tensor
    convergence: torch.Tensor
    step_size: torch.Tensor
    iterations: torch.Tensor
    history: torch.Tensor | None = None


def _single_innovations(system: Model) -> InnovationsStateSpace:
    """Return one innovations system, accepting singleton model batches."""
    iss = _as_innovations_state_space(system)
    parts = []
    for value in (
        iss.transition,
        iss.observation,
        iss.gain,
        iss.innovation_covariance,
    ):
        tensor = torch.as_tensor(value)
        if tensor.ndim == 3:
            if tensor.shape[0] != 1:
                raise ValueError(
                    "DD optimization currently accepts one microscopic system; "
                    "system batching is outside the present parity scope"
                )
            tensor = tensor[0]
        if tensor.ndim != 2:
            raise ValueError("innovations-system matrices must be two-dimensional")
        parts.append(tensor)
    return InnovationsStateSpace(*parts)


def _row_bases(projection: torch.Tensor, n: int) -> tuple[torch.Tensor, bool]:
    """Validate row projections and return shape ``(runs,m,n)``."""
    matrix = torch.as_tensor(projection)
    single = matrix.ndim == 2
    if single:
        matrix = matrix.unsqueeze(0)
    if matrix.ndim != 3:
        raise ValueError("projection must have shape (m,n) or (runs,m,n)")
    if matrix.shape[-1] != n:
        raise ValueError("projection input dimension must match observation dimension")
    if not 1 <= matrix.shape[-2] <= n:
        raise ValueError("projection output dimension must be between 1 and n")
    if matrix.shape[0] < 1:
        raise ValueError("projection must contain at least one run")
    if not bool(torch.isfinite(matrix).all().item()):
        raise ValueError("projection must contain only finite values")
    return matrix, single


def orthonormalise_projection(projection: torch.Tensor) -> torch.Tensor:
    """SVD retraction matching ComplexBox ``orthonormalise``.

    ComplexBox orthonormalises a column basis ``L`` with the left singular
    vectors of ``L``. For ``M=L.T`` this is equivalent to the right singular
    vectors of ``M``.
    """
    matrix = torch.as_tensor(projection)
    single = matrix.ndim == 2
    if single:
        matrix = matrix.unsqueeze(0)
    if matrix.ndim != 3:
        raise ValueError("projection must be unbatched or batched")
    _, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    return vh[0] if single else vh


def _whiten_innovations(
    system: InnovationsStateSpace,
) -> tuple[InnovationsStateSpace, torch.Tensor, bool]:
    r"""Whiten a general innovations system without changing its process.

    Let ``V = B B^T`` be the lower Cholesky factorization and define

    .. math::

       z_t=B^{-1}y_t,\qquad \eta_t=B^{-1}\varepsilon_t.

    Then ``cov(eta)=I`` and the equivalent innovations model is

    .. math::

       x_{t+1}=A x_t + K B\eta_t,\qquad
       z_t=B^{-1}C x_t+\eta_t.

    Thus ``C_w=B^{-1}C`` and ``K_w=KB``. ``B`` is returned because it is
    also the exact coordinate map between physical and whitened projections.
    """
    covariance = torch.as_tensor(system.innovation_covariance)
    identity = torch.eye(
        covariance.shape[-1], dtype=covariance.dtype, device=covariance.device
    )
    is_identity = bool(torch.allclose(covariance, identity, rtol=1e-7, atol=1e-9))
    if is_identity:
        return system, identity, True
    try:
        factor = torch.linalg.cholesky(covariance)
    except RuntimeError as exc:
        raise ValueError("innovation covariance must be positive definite") from exc
    observation = torch.linalg.solve_triangular(
        factor, system.observation, upper=False
    )
    gain = system.gain @ factor
    whitened = InnovationsStateSpace(
        system.transition,
        observation,
        gain,
        identity,
    )
    return whitened, factor, False


def _projection_to_whitened(
    projection: torch.Tensor,
    factor: torch.Tensor,
    *,
    identity_coordinates: bool,
) -> torch.Tensor:
    r"""Map ``M y`` to an orthonormal basis of the same macroprocess in ``z``.

    Since ``y=Bz``, the physical projection ``M`` becomes ``M B`` in whitened
    coordinates. Orthonormalization changes only the macro coordinate basis,
    not the Grassmann subspace or dynamical dependence.
    """
    matrix = torch.as_tensor(
        projection, dtype=factor.dtype, device=factor.device
    )
    if identity_coordinates:
        return matrix
    return orthonormalise_projection(matrix @ factor)


def _projection_from_whitened(
    projection: torch.Tensor,
    factor: torch.Tensor,
    *,
    identity_coordinates: bool,
) -> torch.Tensor:
    r"""Map a whitened row subspace back to the original observation space.

    For ``y=Bz`` and whitened projection ``N z``, the corresponding physical
    row map is ``N B^{-1} y``. The right division is evaluated by a triangular
    solve; no explicit inverse is formed. The returned rows are orthonormalized
    only to choose a Stiefel representative of the same subspace.
    """
    matrix = torch.as_tensor(projection)
    if identity_coordinates:
        return matrix
    single = matrix.ndim == 2
    if single:
        matrix = matrix.unsqueeze(0)
    physical_t = torch.linalg.solve_triangular(
        factor.transpose(-1, -2),
        matrix.transpose(-1, -2),
        upper=True,
    )
    physical = orthonormalise_projection(physical_t.transpose(-1, -2))
    return physical[0] if single else physical


def _restore_result_coordinates(
    result: DDGradientSearchResult,
    factor: torch.Tensor,
    *,
    identity_coordinates: bool,
) -> DDGradientSearchResult:
    """Return an optimizer result with projections in physical coordinates."""
    projection = _projection_from_whitened(
        result.projection,
        factor,
        identity_coordinates=identity_coordinates,
    )
    return DDGradientSearchResult(
        objective=result.objective,
        projection=projection,
        convergence=result.convergence,
        step_size=result.step_size,
        iterations=result.iterations,
        history=result.history,
    )


def innovations_proxy_sequence(system: Model, lags: int | None = None) -> torch.Tensor:
    r"""Return the SSDI proxy sequence ``Q_k = C A^{k-1} K``.

    Returns
    -------
    torch.Tensor
        Shape ``(lags, n, n)``. If omitted, ``lags`` is the latent state
        dimension, matching ComplexBox/SSDI ``iss2cak``.
    """
    iss = _single_innovations(system)
    a, c, k = iss.transition, iss.observation, iss.gain
    state_dimension = a.shape[-1]
    lags = state_dimension if lags is None else int(lags)
    if lags < 1:
        raise ValueError("lags must be at least 1")
    power = torch.eye(state_dimension, dtype=a.dtype, device=a.device)
    values = []
    for lag in range(lags):
        if lag:
            power = power @ a
        values.append(c @ power @ k)
    return torch.stack(values)


def proxy_dynamical_dependence(
    projection: torch.Tensor,
    sequence: torch.Tensor,
) -> torch.Tensor:
    r"""Evaluate ComplexBox ``cak2ddx`` in row-projection convention.

    For ``M=L.T`` and ``Q_k=C A^{k-1}K``,

    .. math::

       D_x(M)=\sum_k \|M Q_k\|_F^2-\|M Q_k M^T\|_F^2.

    This low-level kernel is the identity-innovation ComplexBox objective.
    Model-level wrappers whiten a general ``V > 0`` before calling it.
    """
    q = torch.as_tensor(sequence)
    if q.ndim != 3 or q.shape[-1] != q.shape[-2]:
        raise ValueError("sequence must have shape (lags,n,n)")
    m, single = _row_bases(projection, q.shape[-1])
    q = q.to(dtype=m.dtype, device=m.device)
    mq = torch.einsum("bmi,kij->bkmj", m, q)
    mqmt = torch.einsum("bkmi,bni->bkmn", mq, m)
    value = torch.sum(mq * mq, dim=(1, 2, 3)) - torch.sum(
        mqmt * mqmt, dim=(1, 2, 3)
    )
    return value[0] if single else value


def proxy_dynamical_dependence_gradient(
    projection: torch.Tensor,
    sequence: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Return the analytic Grassmannian gradient of proxy DD.

    ComplexBox uses

    .. math::

       g=\sum_k(Q_kQ_k^T-Q_k^TPQ_k-Q_kPQ_k^T),\qquad
       G=2gL-P(2gL),\quad P=LL^T.

    The result here is ``G.T`` because ComplexTorch stores ``M=L.T``.
    """
    q = torch.as_tensor(sequence)
    if q.ndim != 3 or q.shape[-1] != q.shape[-2]:
        raise ValueError("sequence must have shape (lags,n,n)")
    m, single = _row_bases(projection, q.shape[-1])
    q = q.to(dtype=m.dtype, device=m.device)
    l = m.transpose(-1, -2)
    p = l @ l.transpose(-1, -2)
    qqt = torch.sum(q @ q.transpose(-1, -2), dim=0)
    raw = qqt.unsqueeze(0) @ l
    for qk in q:
        raw = raw - qk.transpose(-1, -2) @ p @ qk @ l
        raw = raw - qk @ p @ qk.transpose(-1, -2) @ l
    grad_l = 2.0 * raw
    grad_l = grad_l - p @ grad_l
    grad_m = grad_l.transpose(-1, -2)
    magnitude = torch.linalg.vector_norm(grad_m, dim=(-2, -1))
    if single:
        return grad_m[0], magnitude[0]
    return grad_m, magnitude


def spectral_dynamical_dependence(
    projection: torch.Tensor,
    transfer: torch.Tensor,
) -> torch.Tensor:
    r"""Evaluate ComplexBox ``trfun2dd`` for identity innovations.

    ``transfer`` has shape ``(frequencies,n,n)``. The one-sided trapezoidal
    normalisation is exactly the convention used by ComplexBox/SSDI.
    """
    h = torch.as_tensor(transfer)
    if h.ndim != 3 or h.shape[-1] != h.shape[-2] or h.shape[0] < 2:
        raise ValueError(
            "transfer must have shape (frequencies,n,n), frequencies>=2"
        )
    m, single = _row_bases(projection, h.shape[-1])
    real_dtype = m.dtype
    complex_dtype = (
        torch.complex128 if real_dtype == torch.float64 else torch.complex64
    )
    h = h.to(dtype=complex_dtype, device=m.device)
    l = m.transpose(-1, -2).to(complex_dtype)
    hl = torch.einsum("fij,bjm->bfim", h.conj().transpose(-1, -2), l)
    gram = torch.einsum("bfim,bfin->bfmn", hl.conj(), hl)
    gram = 0.5 * (gram + gram.conj().transpose(-1, -2))
    chol = torch.linalg.cholesky(gram)
    diagonal = torch.real(torch.diagonal(chol, dim1=-2, dim2=-1))
    logdet = 2.0 * torch.sum(torch.log(diagonal), dim=-1)
    weights = torch.ones(h.shape[0], dtype=real_dtype, device=m.device)
    weights[0] = 0.5
    weights[-1] = 0.5
    weights = weights / float(h.shape[0] - 1)
    value = torch.sum(logdet * weights.unsqueeze(0), dim=-1)
    return value[0] if single else value


def spectral_dynamical_dependence_gradient(
    projection: torch.Tensor,
    transfer: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Return ComplexBox/SSDI's analytic spectral Grassmannian gradient.

    This ports Appendix-D/``trfun2ddgrad`` in the row-projection convention.
    Cholesky solves replace explicit inverses without changing the equation.
    """
    h = torch.as_tensor(transfer)
    if h.ndim != 3 or h.shape[-1] != h.shape[-2] or h.shape[0] < 2:
        raise ValueError(
            "transfer must have shape (frequencies,n,n), frequencies>=2"
        )
    m, single = _row_bases(projection, h.shape[-1])
    real_dtype = m.dtype
    complex_dtype = (
        torch.complex128 if real_dtype == torch.float64 else torch.complex64
    )
    h = h.to(dtype=complex_dtype, device=m.device)
    l = m.transpose(-1, -2).to(complex_dtype)
    weights = torch.ones(h.shape[0], dtype=real_dtype, device=m.device)
    weights[0] = 0.5
    weights[-1] = 0.5
    weights = weights / float(h.shape[0] - 1)
    grad_l = torch.zeros_like(l.real)
    for index, hk in enumerate(h):
        hl = hk.conj().transpose(-1, -2).unsqueeze(0) @ l
        gram = hl.conj().transpose(-1, -2) @ hl
        gram = 0.5 * (gram + gram.conj().transpose(-1, -2))
        chol = torch.linalg.cholesky(gram)
        numerator = hk.unsqueeze(0) @ hl
        solved = torch.cholesky_solve(
            numerator.conj().transpose(-1, -2), chol
        )
        right_divided = solved.conj().transpose(-1, -2)
        grad_l = grad_l + 2.0 * weights[index] * right_divided.real
    grad_l = grad_l - 2.0 * l.real
    grad_m = grad_l.transpose(-1, -2)
    magnitude = torch.linalg.vector_norm(grad_m, dim=(-2, -1))
    if single:
        return grad_m[0], magnitude[0]
    return grad_m, magnitude


def _parse_factors(gdls: float | tuple[float, float]) -> tuple[float, float]:
    """Parse ComplexBox's gradient-descent line-search factors."""
    if isinstance(gdls, (float, int)):
        ifac = float(gdls)
        if ifac <= 0.0:
            raise ValueError("gdls must be positive")
        return ifac, 1.0 / ifac
    if len(gdls) != 2:
        raise ValueError("gdls must be a scalar or a two-element tuple")
    ifac, nfac = map(float, gdls)
    if ifac <= 0.0 or nfac <= 0.0:
        raise ValueError("gdls factors must be positive")
    return ifac, nfac


def _parse_tolerance(
    tol: float | tuple[float, float, float], *, spectral: bool
) -> tuple[float, float, float]:
    """Parse SSDI's ``(step, objective, gradient)`` tolerances."""
    if isinstance(tol, (float, int)):
        value = float(tol)
        return value, value, value if spectral else value / 10.0
    if len(tol) != 3:
        raise ValueError("tol must be a scalar or a three-element tuple")
    return tuple(float(value) for value in tol)


def _optimise(
    initial_projection: torch.Tensor,
    *,
    objective: Callable[[torch.Tensor], torch.Tensor],
    gradient: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    max_iterations: int,
    variant: int,
    initial_step_size: float,
    gdls: float | tuple[float, float],
    tol: float | tuple[float, float, float],
    spectral: bool,
    history: bool,
) -> DDGradientSearchResult:
    """Batched port of ComplexBox ``_optimise_tensor_batch``."""
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    if variant not in (1, 2):
        raise ValueError("variant must be 1 or 2")
    if initial_step_size <= 0.0:
        raise ValueError("initial_step_size must be positive")
    ifac, nfac = _parse_factors(gdls)
    stol, dtol, gtol = _parse_tolerance(tol, spectral=spectral)

    projection = torch.as_tensor(initial_projection)
    if projection.ndim == 2:
        projection = projection.unsqueeze(0)
    if projection.ndim != 3:
        raise ValueError("initial_projection must have shape (m,n) or (runs,m,n)")
    l = projection.clone()
    nruns = l.shape[0]
    dd = objective(l)
    grad, gmag = gradient(l)
    sigma = torch.full(
        (nruns,), float(initial_step_size), dtype=l.dtype, device=l.device
    )
    active = torch.ones(nruns, dtype=torch.bool, device=l.device)
    convergence = torch.zeros(nruns, dtype=torch.int64, device=l.device)
    stop_iterations = torch.full(
        (nruns,), int(max_iterations), dtype=torch.int64, device=l.device
    )

    hist = None
    if history:
        hist = torch.full(
            (nruns, max_iterations, 3),
            torch.nan,
            dtype=l.dtype,
            device=l.device,
        )
        hist[:, 0, :] = torch.stack((dd, sigma, gmag), dim=-1)

    with torch.no_grad():
        for iteration in range(2, max_iterations + 1):
            safe = torch.where(gmag > 0.0, gmag, torch.ones_like(gmag))
            step = sigma[:, None, None] * grad / safe[:, None, None]
            step = torch.where(
                active[:, None, None], step, torch.zeros_like(step)
            )
            candidate_all = orthonormalise_projection(l - step)
            candidate = torch.where(active[:, None, None], candidate_all, l)

            if variant == 1:
                dd_try = objective(candidate)
                accept = active & (dd_try < dd)
                l = torch.where(accept[:, None, None], candidate, l)
                dd = torch.where(accept, dd_try, dd)
                sigma = torch.where(
                    active,
                    torch.where(accept, sigma * ifac, sigma * nfac),
                    sigma,
                )
                grad_try, gmag_try = gradient(l)
                grad = torch.where(accept[:, None, None], grad_try, grad)
                gmag = torch.where(accept, gmag_try, gmag)
            else:
                l = candidate
                grad_new, gmag_new = gradient(l)
                dd_new = objective(l)
                improve = active & (dd_new < dd)
                dd = torch.where(improve, dd_new, dd)
                sigma = torch.where(
                    active,
                    torch.where(improve, sigma * ifac, sigma * nfac),
                    sigma,
                )
                grad = torch.where(active[:, None, None], grad_new, grad)
                gmag = torch.where(active, gmag_new, gmag)

            if hist is not None:
                hist[:, iteration - 1, :] = torch.stack(
                    (dd, sigma, gmag), dim=-1
                )

            c1 = active & (sigma < stol)
            remaining = active & ~c1
            c2 = remaining & (dd < dtol)
            remaining = remaining & ~c2
            c3 = remaining & (gmag < gtol)
            stopped = c1 | c2 | c3
            convergence = torch.where(
                c1, torch.ones_like(convergence), convergence
            )
            convergence = torch.where(
                c2, torch.full_like(convergence, 2), convergence
            )
            convergence = torch.where(
                c3, torch.full_like(convergence, 3), convergence
            )
            stop_iterations = torch.where(
                stopped,
                torch.full_like(stop_iterations, iteration),
                stop_iterations,
            )
            active = active & ~stopped
            if not bool(torch.any(active).item()):
                break

    order = torch.argsort(dd, stable=True)
    return DDGradientSearchResult(
        objective=dd[order],
        projection=l[order],
        convergence=convergence[order],
        step_size=sigma[order],
        iterations=stop_iterations[order],
        history=None if hist is None else hist[order],
    )


def optimise_dynamical_dependence_proxy(
    system: Model,
    initial_projection: torch.Tensor,
    *,
    lags: int | None = None,
    max_iterations: int = 10_000,
    variant: int = 2,
    initial_step_size: float = 1e-3,
    gdls: float | tuple[float, float] = 2.0,
    tol: float | tuple[float, float, float] = 1e-9,
    history: bool = False,
) -> DDGradientSearchResult:
    r"""Optimize ComplexBox proxy DD for any positive-definite ``V``.

    General innovations are whitened exactly using ``V=B B^T``. If ``M`` is
    an initial physical projection, the optimizer is initialized on the
    equivalent whitened subspace ``row(MB)`` and the optimized subspace is
    mapped back through ``B^{-1}`` before being returned.
    """
    iss = _single_innovations(system)
    whitened, factor, identity_coordinates = _whiten_innovations(iss)
    sequence = innovations_proxy_sequence(whitened, lags=lags)
    initial = _projection_to_whitened(
        initial_projection,
        factor,
        identity_coordinates=identity_coordinates,
    ).to(dtype=sequence.dtype, device=sequence.device)
    result = _optimise(
        initial,
        objective=lambda matrix: proxy_dynamical_dependence(matrix, sequence),
        gradient=lambda matrix: proxy_dynamical_dependence_gradient(
            matrix, sequence
        ),
        max_iterations=max_iterations,
        variant=variant,
        initial_step_size=initial_step_size,
        gdls=gdls,
        tol=tol,
        spectral=False,
        history=history,
    )
    return _restore_result_coordinates(
        result,
        factor,
        identity_coordinates=identity_coordinates,
    )


def optimise_dynamical_dependence_spectral(
    system: Model,
    initial_projection: torch.Tensor,
    frequencies: torch.Tensor,
    *,
    max_iterations: int = 10_000,
    variant: int = 2,
    initial_step_size: float = 1e-3,
    gdls: float | tuple[float, float] = 2.0,
    tol: float | tuple[float, float, float] = 1e-9,
    history: bool = False,
) -> DDGradientSearchResult:
    r"""Optimize ComplexBox spectral DD for any positive-definite ``V``.

    The general ISS is transformed to the exactly equivalent identity-
    innovation coordinates before applying ``trfun2dd``/``trfun2ddgrad``.
    Returned projections are mapped back to the original observation space.
    """
    iss = _single_innovations(system)
    whitened, factor, identity_coordinates = _whiten_innovations(iss)
    frequencies = torch.as_tensor(
        frequencies,
        dtype=whitened.transition.dtype,
        device=whitened.transition.device,
    )
    transfer = innovations_transfer_function(whitened, frequencies)
    if transfer.ndim == 4:
        transfer = transfer[0]
    initial = _projection_to_whitened(
        initial_projection,
        factor,
        identity_coordinates=identity_coordinates,
    ).to(dtype=whitened.observation.dtype, device=whitened.observation.device)
    result = _optimise(
        initial,
        objective=lambda matrix: spectral_dynamical_dependence(
            matrix, transfer
        ),
        gradient=lambda matrix: spectral_dynamical_dependence_gradient(
            matrix, transfer
        ),
        max_iterations=max_iterations,
        variant=variant,
        initial_step_size=initial_step_size,
        gdls=gdls,
        tol=tol,
        spectral=True,
        history=history,
    )
    return _restore_result_coordinates(
        result,
        factor,
        identity_coordinates=identity_coordinates,
    )


__all__ = [
    "DDGradientSearchResult",
    "innovations_proxy_sequence",
    "orthonormalise_projection",
    "proxy_dynamical_dependence",
    "proxy_dynamical_dependence_gradient",
    "spectral_dynamical_dependence",
    "spectral_dynamical_dependence_gradient",
    "optimise_dynamical_dependence_proxy",
    "optimise_dynamical_dependence_spectral",
]
