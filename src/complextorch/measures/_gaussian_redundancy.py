"""Shared Gaussian redundancy primitives for PID and PhiID.

This private module contains the covariance-only numerical primitives shared by
two-source Gaussian PID and bivariate Gaussian PhiID. Keeping one implementation
prevents the two decompositions from drifting while leaving their lattice and
public-API logic separate. CCS expectations use deterministic Sobol
quasi-Monte-Carlo integration; dependency-constraint redundancy follows the
Gaussian Kay--Ince construction.

References
----------
- Ince, R. A. A. (2017). Measuring multivariate redundant information with
  pointwise common change in surprisal. *Entropy*, 19, 318.
- Kay, J. W. and Ince, R. A. A. (2018). Exact partial information
  decompositions for Gaussian systems based on dependency constraints.
- ``robince/partial-info-decomp`` commit
  ``32207164741b9e3ba86cec225c09b4b617681e93``.
"""
from __future__ import annotations

import math
from typing import Literal

import torch

from ..linalg import spd_logdet
from .gaussian import gaussian_mutual_information

GaussianRedundancy = Literal["mmi", "ccs", "idep_a", "idep_b"]


def _index(indices, covariance: torch.Tensor) -> torch.Tensor:
    """Build a device-local integer index tensor for covariance blocks."""
    return torch.as_tensor(tuple(indices), dtype=torch.long, device=covariance.device)


def _subcov(covariance: torch.Tensor, indices) -> torch.Tensor:
    """Extract a square covariance submatrix while preserving batch dimensions."""
    index = _index(indices, covariance)
    return covariance.index_select(-2, index).index_select(-1, index)


def _mi(covariance: torch.Tensor, left, right, *, base: float) -> torch.Tensor:
    """Evaluate Gaussian mutual information between selected covariance blocks."""
    left = tuple(left)
    right = tuple(right)
    block = _subcov(covariance, left + right)
    return gaussian_mutual_information(block, len(left), base=base)


def _sobol_gaussian_samples(covariance: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Deterministically integrate under N(0, covariance) without data fitting."""
    if n_samples < 32:
        raise ValueError("ccs_qmc_samples must be at least 32")
    dimension = covariance.shape[-1]
    engine = torch.quasirandom.SobolEngine(dimension, scramble=False)
    # The first Sobol point is exactly zero and maps to -inf under Phi^-1.
    uniform = engine.draw(n_samples + 1, dtype=covariance.dtype)[1:].to(covariance.device)
    eps = torch.finfo(covariance.dtype).eps
    uniform = uniform.clamp(min=eps, max=1.0 - eps)
    standard_normal = math.sqrt(2.0) * torch.erfinv(2.0 * uniform - 1.0)
    factor = torch.linalg.cholesky(covariance)
    return torch.einsum("nd,...kd->...nk", standard_normal, factor)


def _quadratic_form(samples: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
    """Evaluate batched Gaussian quadratic forms using Cholesky solves."""
    factor = torch.linalg.cholesky(covariance)
    rhs = samples.transpose(-1, -2)
    solved = torch.cholesky_solve(rhs, factor).transpose(-1, -2)
    return (samples * solved).sum(-1)


def _local_mi(
    samples: torch.Tensor,
    covariance: torch.Tensor,
    left,
    right,
    *,
    base: float,
) -> torch.Tensor:
    """Pointwise Gaussian MI evaluated on common model-integration nodes."""
    left = tuple(left)
    right = tuple(right)
    indices = left + right
    index = _index(indices, covariance)
    values = samples.index_select(-1, index)
    block = _subcov(covariance, indices)
    n_left = len(left)
    left_values = values[..., :n_left]
    right_values = values[..., n_left:]
    left_cov = block[..., :n_left, :n_left]
    right_cov = block[..., n_left:, n_left:]
    logdet_term = 0.5 * (
        spd_logdet(left_cov) + spd_logdet(right_cov) - spd_logdet(block)
    )
    local = logdet_term.unsqueeze(-1) - 0.5 * (
        _quadratic_form(values, block)
        - _quadratic_form(left_values, left_cov)
        - _quadratic_form(right_values, right_cov)
    )
    return local / math.log(base)


def _ccs_local(
    samples: torch.Tensor,
    covariance: torch.Tensor,
    source0,
    source1,
    target,
    *,
    base: float,
) -> torch.Tensor:
    """Ince CCS local redundancy with the reference sign-coherence rule."""
    source0 = tuple(source0)
    source1 = tuple(source1)
    target = tuple(target)
    mi0 = _local_mi(samples, covariance, source0, target, base=base)
    mi1 = _local_mi(samples, covariance, source1, target, base=base)
    joint = _local_mi(samples, covariance, source0 + source1, target, base=base)
    common = mi0 + mi1 - joint
    signs = torch.stack(
        [torch.sign(mi0), torch.sign(mi1), torch.sign(joint), torch.sign(common)],
        dim=-1,
    )
    coherent = (signs == signs[..., :1]).all(-1)
    return torch.where(coherent, common, torch.zeros_like(common))


def _ccs(
    samples: torch.Tensor,
    covariance: torch.Tensor,
    source0,
    source1,
    target,
    *,
    base: float,
) -> torch.Tensor:
    """Average local CCS redundancy over deterministic model-integration nodes."""
    return _ccs_local(
        samples, covariance, source0, source1, target, base=base
    ).mean(-1)


def _whitened_cross(covariance: torch.Tensor, left, right) -> torch.Tensor:
    """Return the blockwise-whitened cross covariance Lx^-1 Cxy Ly^-T."""
    left = tuple(left)
    right = tuple(right)
    left_index = _index(left, covariance)
    right_index = _index(right, covariance)
    cross = covariance.index_select(-2, left_index).index_select(-1, right_index)
    left_factor = torch.linalg.cholesky(_subcov(covariance, left))
    right_factor = torch.linalg.cholesky(_subcov(covariance, right))
    first = torch.linalg.solve_triangular(left_factor, cross, upper=False)
    return torch.linalg.solve_triangular(
        right_factor, first.transpose(-1, -2), upper=False
    ).transpose(-1, -2)


def _identity(covariance: torch.Tensor, dimension: int) -> torch.Tensor:
    """Create a batch-expanded identity matrix matching covariance dtype/device."""
    eye = torch.eye(dimension, dtype=covariance.dtype, device=covariance.device)
    return eye.expand(*covariance.shape[:-2], dimension, dimension)


def _half_logdet(matrix: torch.Tensor, *, base: float) -> torch.Tensor:
    """Evaluate one-half of an SPD log-determinant in the requested log base."""
    return 0.5 * spd_logdet(matrix) / math.log(base)


def _idep(
    covariance: torch.Tensor,
    source0,
    source1,
    target,
    *,
    base: float,
) -> torch.Tensor:
    """Gaussian two-predictor I_dep redundancy (Kay-Ince Table 9)."""
    source0 = tuple(source0)
    source1 = tuple(source1)
    target = tuple(target)
    p = _whitened_cross(covariance, source0, source1)
    q = _whitened_cross(covariance, source0, target)
    r = _whitened_cross(covariance, source1, target)
    ex = _identity(covariance, len(source0))
    ey = _identity(covariance, len(source1))
    ez = _identity(covariance, len(target))

    ix = _mi(covariance, source0, target, base=base)
    iy = _mi(covariance, source1, target, base=base)

    # Dependency-lattice edges b, i, and k from Kay & Ince (2018), Table 9.
    b_edge = ix
    rq = r @ q.transpose(-1, -2)
    i_edge = (
        _half_logdet(ey - rq @ rq.transpose(-1, -2), base=base)
        - _half_logdet(ez - q.transpose(-1, -2) @ q, base=base)
        - _half_logdet(ez - r.transpose(-1, -2) @ r, base=base)
        - iy
    )
    standardized = torch.cat(
        [
            torch.cat([ex, p, q], dim=-1),
            torch.cat([p.transpose(-1, -2), ey, r], dim=-1),
            torch.cat([q.transpose(-1, -2), r.transpose(-1, -2), ez], dim=-1),
        ],
        dim=-2,
    )
    k_edge = (
        _half_logdet(ey - p.transpose(-1, -2) @ p, base=base)
        - _half_logdet(standardized, base=base)
        - iy
    )
    unique0 = torch.stack([b_edge, i_edge, k_edge], dim=-1).amin(-1)
    redundancy = ix - unique0
    # Remove only floating-point roundoff around the theoretical zero boundary.
    tolerance = 64.0 * torch.finfo(covariance.dtype).eps
    return torch.where(
        (redundancy < 0) & (redundancy >= -tolerance),
        torch.zeros_like(redundancy),
        redundancy,
    )


def _single_target_redundancy(
    covariance: torch.Tensor,
    source0,
    source1,
    target,
    *,
    redundancy: GaussianRedundancy,
    base: float,
    ccs_samples: torch.Tensor | None,
) -> torch.Tensor:
    """Dispatch one Gaussian two-source redundancy to the selected backend."""
    if redundancy == "mmi":
        return torch.minimum(
            _mi(covariance, source0, target, base=base),
            _mi(covariance, source1, target, base=base),
        )
    if redundancy == "ccs":
        if ccs_samples is None:
            raise RuntimeError("CCS integration nodes were not initialized")
        return _ccs(
            ccs_samples, covariance, source0, source1, target, base=base
        )
    return _idep(covariance, source0, source1, target, base=base)


__all__: list[str] = []
