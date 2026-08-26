r"""Mechanistic modal primitives for stationary linear Gaussian systems.

The common innovations representation

.. math::

   x_{t+1}=Ax_t+K\varepsilon_t,\qquad
   y_t=Cx_t+\varepsilon_t,\qquad
   \operatorname{cov}(\varepsilon_t)=V

has transfer function

.. math::

   H(z)=I+C(zI-A)^{-1}K.

For a diagonalizable transition with simple poles, the strictly proper part has
modal residues ``R_j`` such that

.. math::

   H(z)=I+\sum_j \frac{R_j}{z-\lambda_j}.

These poles and residues are process-level objects: they are invariant to
latent-state similarity transforms even though the state-space matrices are
not.  This module exposes that representation without introducing a second
measure implementation.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .representations import StateSpaceModel, VARSystem
from .control import InnovationsStateSpace
from .transformations import as_innovations_state_space

ModelSystem = StateSpaceModel | InnovationsStateSpace | VARSystem


@dataclass(frozen=True)
class ModalDecomposition:
    r"""Pole--residue representation of an innovations-form process.

    Attributes
    ----------
    poles
        Transition eigenvalues with shape ``(..., r)``.
    residues
        Transfer-function residues with shape ``(..., r, n, n)``.
    strengths
        Largest singular value of each residue, shape ``(..., r)``.
    normalized_residues
        ``residues / strengths`` where the strength is nonzero.
    active
        Boolean mask for modes with a numerically nonzero transfer residue.
        Distinct nonminimal state modes are retained with ``active=False``.
    innovation_covariance
        Innovation covariance ``V`` in observation coordinates.
    eigenvector_condition
        Condition number of the right-eigenvector matrix. Large values signal
        that individual simple-mode residues are numerically sensitive.
    """

    poles: torch.Tensor
    residues: torch.Tensor
    strengths: torch.Tensor
    normalized_residues: torch.Tensor
    active: torch.Tensor
    innovation_covariance: torch.Tensor
    eigenvector_condition: torch.Tensor

    @property
    def normalized_innovation_covariance(self) -> torch.Tensor:
        """Return ``V / trace(V)``, removing the analytically irrelevant global scale."""
        covariance = self.innovation_covariance
        trace = torch.diagonal(covariance, dim1=-2, dim2=-1).sum(-1)
        return covariance / trace[..., None, None]


def _batched_matrix(value: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Normalize a system matrix to one leading batch dimension."""
    tensor = torch.as_tensor(value)
    single = tensor.ndim == 2
    if single:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError("system matrices must be unbatched or singly batched")
    return tensor, single


def modal_decomposition(
    system: ModelSystem,
    *,
    simple_pole_tolerance: float | None = None,
    residue_tolerance: float | None = None,
) -> ModalDecomposition:
    r"""Return poles, transfer residues, and residue strengths.

    For ``A = X diag(lambda) X^-1``, the resolvent projector of mode ``j`` is
    ``P_j = x_j w_j^*``, where ``w_j^*`` is row ``j`` of ``X^-1``.  Therefore

    .. math::

       R_j = C P_j K = (C x_j)(w_j^* K).

    The solve for ``X^-1`` is performed with :func:`torch.linalg.solve`; no
    explicit matrix inverse is formed.  The calculation is fully batched.

    Parameters
    ----------
    system
        Canonical VAR, general state-space, or innovations-form process.
    simple_pole_tolerance
        Minimum absolute separation required between distinct poles. If
        omitted, a dtype-scaled tolerance is used. Individual modal residues
        are not unique for repeated poles, so such systems are rejected.
    residue_tolerance
        Absolute threshold used to mark transfer-active modes. If omitted, a
        dtype-scaled threshold relative to the largest residue strength is used.

    Returns
    -------
    ModalDecomposition
        Batched or unbatched process-level modal representation matching the
        input batch semantics.

    Notes
    -----
    The decomposition assumes a diagonalizable transition with simple poles.
    Near defective transitions can have a large ``eigenvector_condition``;
    that sensitivity is retained rather than silently regularized. Distinct
    state modes with zero transfer residue are retained and flagged inactive,
    so nonminimal augmentation does not masquerade as an observable mechanism.

    References
    ----------
    - Kailath, T. (1980). *Linear Systems*. Prentice-Hall.
    - Trefethen, L. N. and Embree, M. (2005). *Spectra and Pseudospectra*.
    """
    innovations = as_innovations_state_space(system)
    a, a_single = _batched_matrix(innovations.transition)
    c, c_single = _batched_matrix(innovations.observation)
    k, k_single = _batched_matrix(innovations.gain)
    v, v_single = _batched_matrix(innovations.innovation_covariance)

    if not all(t.is_floating_point() for t in (a, c, k, v)):
        raise TypeError("system matrices must use floating-point dtypes")
    if len({t.dtype for t in (a, c, k, v)}) != 1:
        raise ValueError("system matrices must share dtype")
    if len({t.device for t in (a, c, k, v)}) != 1:
        raise ValueError("system matrices must share device")
    if a.shape[-1] != a.shape[-2]:
        raise ValueError("transition must be square")
    if c.shape[-1] != a.shape[-1] or k.shape[-2] != a.shape[-1]:
        raise ValueError("incompatible transition, observation, or gain dimensions")
    if c.shape[-2] != k.shape[-1] or v.shape[-1] != c.shape[-2] or v.shape[-2] != c.shape[-2]:
        raise ValueError("innovation covariance must match observation dimension")

    batch = max(t.shape[0] for t in (a, c, k, v))
    if any(t.shape[0] not in (1, batch) for t in (a, c, k, v)):
        raise ValueError("incompatible system batch dimensions")
    a, c, k, v = [
        t.expand(batch, *t.shape[1:]) if t.shape[0] == 1 else t
        for t in (a, c, k, v)
    ]

    poles, right = torch.linalg.eig(a)
    state_dimension = poles.shape[-1]
    if state_dimension > 1:
        separation = torch.abs(poles[..., :, None] - poles[..., None, :])
        eye = torch.eye(state_dimension, dtype=torch.bool, device=poles.device)
        separation = separation.masked_fill(eye, torch.inf)
        minimum_separation = separation.amin(dim=(-2, -1))
        if simple_pole_tolerance is None:
            real_dtype = a.dtype
            scale = poles.abs().amax(-1).clamp_min(torch.ones((), dtype=real_dtype, device=a.device))
            tolerance = 100.0 * torch.finfo(real_dtype).eps * scale
        else:
            if simple_pole_tolerance < 0:
                raise ValueError("simple_pole_tolerance must be non-negative")
            tolerance = torch.full_like(minimum_separation.real, float(simple_pole_tolerance))
        if bool(torch.any(minimum_separation <= tolerance).item()):
            raise ValueError("modal_decomposition requires simple, separated poles")

    complex_dtype = right.dtype
    identity = torch.eye(state_dimension, dtype=complex_dtype, device=a.device).expand(batch, -1, -1)
    left_rows = torch.linalg.solve(right, identity)
    c_complex = c.to(complex_dtype)
    k_complex = k.to(complex_dtype)

    output_vectors = c_complex @ right
    input_rows = left_rows @ k_complex
    residues = output_vectors.transpose(-1, -2)[..., :, :, None] * input_rows[..., :, None, :]
    strengths = torch.linalg.svdvals(residues)[..., 0].real
    tiny = torch.finfo(strengths.dtype).tiny
    if residue_tolerance is None:
        strength_scale = strengths.amax(-1, keepdim=True).clamp_min(1.0)
        active_threshold = 100.0 * torch.finfo(strengths.dtype).eps * strength_scale
    else:
        if residue_tolerance < 0:
            raise ValueError("residue_tolerance must be non-negative")
        active_threshold = torch.full_like(strengths[..., :1], float(residue_tolerance))
    active = strengths > active_threshold
    safe_strength = strengths.clamp_min(tiny)
    normalized = residues / safe_strength[..., None, None]
    normalized = torch.where(
        active[..., None, None], normalized, torch.zeros_like(normalized)
    )
    condition = torch.linalg.cond(right)

    single = a_single and c_single and k_single and v_single
    if single:
        return ModalDecomposition(
            poles[0],
            residues[0],
            strengths[0],
            normalized[0],
            active[0],
            v[0],
            condition[0],
        )
    return ModalDecomposition(
        poles, residues, strengths, normalized, active, v, condition
    )


def modal_observation_covariance(
    decomposition: ModalDecomposition,
) -> torch.Tensor:
    r"""Reconstruct the stationary observation covariance from modal objects.

    For a stable innovations process with simple poles,

    .. math::

       \Gamma_0 = V + \sum_{j,k}
       \frac{R_j V R_k^*}{1-\lambda_j\bar\lambda_k}.

    This is an exact analytical identity, not a spectral quadrature.  Batched
    dimensions are preserved.
    """
    poles = torch.as_tensor(decomposition.poles)
    residues = torch.as_tensor(decomposition.residues)
    covariance = torch.as_tensor(decomposition.innovation_covariance)
    single = poles.ndim == 1
    if single:
        poles = poles.unsqueeze(0)
        residues = residues.unsqueeze(0)
        covariance = covariance.unsqueeze(0)
    if poles.ndim != 2 or residues.ndim != 4 or covariance.ndim != 3:
        raise ValueError("invalid ModalDecomposition tensor shapes")
    if bool(torch.any(poles.abs() >= 1).item()):
        raise ValueError("modal covariance requires strictly stable poles")

    v = covariance.to(residues.dtype)
    left = residues[:, :, None] @ v[:, None, None]
    terms = left @ residues[:, None].conj().transpose(-1, -2)
    denominator = 1.0 - poles[:, :, None] * poles[:, None].conj()
    dynamic = (terms / denominator[..., None, None]).sum(dim=(1, 2))
    result = v + dynamic
    result = 0.5 * (result + result.conj().transpose(-1, -2))
    if not decomposition.innovation_covariance.is_complex():
        result = result.real
    return result[0] if single else result
