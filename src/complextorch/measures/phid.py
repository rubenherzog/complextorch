"""Gaussian integrated information decomposition (PhiID).

Bivariate PhiID is computed from a supplied joint past/future covariance. The
implementation follows the 16-equation ``15-for-free`` construction used by the
Imperial MIND-lab reference implementation: ordinary mutual informations and
forward/backward PID redundancies determine fifteen cumulative terms and one
additional double-redundancy closes the product lattice.

MMI and Gaussian dependency-constraint PID are analytic covariance functions.
CCS is pointwise, so its Gaussian model expectation is evaluated by a
deterministic Sobol quasi-Monte-Carlo rule under the supplied covariance; no
observations are fitted or used to estimate the covariance.

References
----------
- Mediano, P. A. M. et al. (2021). Towards an extended taxonomy of information
  dynamics via Integrated Information Decomposition.
- Ince, R. A. A. (2017). Measuring multivariate redundant information with
  pointwise common change in surprisal. Entropy 19, 318.
- Kay, J. W. and Ince, R. A. A. (2018). Exact partial information
  decompositions for Gaussian systems based on dependency constraints.
- Imperial MIND-lab ``integrated-info-decomp`` commit
  ``6c5f2e9d33c985efbdf875d45cb5a2a6a5cdbf44``.
- ``robince/partial-info-decomp`` commit
  ``32207164741b9e3ba86cec225c09b4b617681e93``.
"""
from __future__ import annotations

import logging
import math
from typing import Literal

import torch

from ..linalg import spd_logdet
from .gaussian import gaussian_mutual_information

PhiIDRedundancy = Literal["mmi", "ccs", "idep_a", "idep_b"]

ATOM_LABELS = (
    "red_to_red",
    "red_to_unq0",
    "red_to_unq1",
    "red_to_syn",
    "unq0_to_red",
    "unq0_to_unq0",
    "unq0_to_unq1",
    "unq0_to_syn",
    "unq1_to_red",
    "unq1_to_unq0",
    "unq1_to_unq1",
    "unq1_to_syn",
    "syn_to_red",
    "syn_to_unq0",
    "syn_to_unq1",
    "syn_to_syn",
)

# Rows are the cumulative quantities used by the reference implementation:
# rtr, Rxyta, Rxytb, Rxytab, Rabtx, Rabty, Rabtxy, Ixta, Ixtb, Iyta,
# Iytb, Ixyta, Ixytb, Ixtab, Iytab, Ixytab. Columns are ATOM_LABELS.
_KNOWNS_TO_ATOMS = (
    (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
    (1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0),
    (1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0),
    (1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0),
    (1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0),
    (1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0),
    (1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0),
    (1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0),
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
)

_LOG = logging.getLogger(__name__)


def _index(indices, covariance: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(tuple(indices), dtype=torch.long, device=covariance.device)


def _subcov(covariance: torch.Tensor, indices) -> torch.Tensor:
    index = _index(indices, covariance)
    return covariance.index_select(-2, index).index_select(-1, index)


def _mi(
    covariance: torch.Tensor,
    left,
    right,
    *,
    base: float,
) -> torch.Tensor:
    left = tuple(left)
    right = tuple(right)
    block = _subcov(covariance, left + right)
    return gaussian_mutual_information(block, len(left), base=base)


def _validate_covariance(joint_covariance: torch.Tensor, block_size: int) -> torch.Tensor:
    covariance = torch.as_tensor(joint_covariance)
    if not covariance.is_floating_point():
        covariance = covariance.to(torch.get_default_dtype())
    if covariance.ndim < 2 or covariance.shape[-2] != covariance.shape[-1]:
        raise ValueError("joint_covariance must be square on its final two dimensions")
    if block_size < 1 or covariance.shape[-1] != 4 * block_size:
        raise ValueError("covariance dimension must equal 4 * block_size")
    if not bool(torch.isfinite(covariance).all()):
        raise ValueError("joint_covariance must contain only finite values")
    # Fail explicitly if the supplied Gaussian model does not define an SPD law.
    torch.linalg.cholesky(covariance)
    return covariance


def _validate_redundancy(redundancy: str) -> PhiIDRedundancy:
    name = redundancy.lower()
    allowed = ("mmi", "ccs", "idep_a", "idep_b")
    if name not in allowed:
        if name in ("sx", "tau_sx", "i_tau_sx", "itsx", "varley"):
            raise NotImplementedError(
                "Varley's I_tau_sx redundancy is defined for discrete random variables; "
                "the continuous Gaussian primary PhiID path does not silently "
                "discretize the model or invent a Gaussian extension."
            )
        raise ValueError(f"redundancy must be one of {allowed}, got {redundancy!r}")
    return name  # type: ignore[return-value]


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
    eye = torch.eye(dimension, dtype=covariance.dtype, device=covariance.device)
    return eye.expand(*covariance.shape[:-2], dimension, dimension)


def _half_logdet(matrix: torch.Tensor, *, base: float) -> torch.Tensor:
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
    redundancy: PhiIDRedundancy,
    base: float,
    ccs_samples: torch.Tensor | None,
) -> torch.Tensor:
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


def _double_ccs(
    samples: torch.Tensor,
    covariance: torch.Tensor,
    sources,
    targets,
    *,
    base: float,
) -> torch.Tensor:
    """Exact Imperial MIND-lab local double-CCS construction."""
    x, y = sources
    a, b = targets
    i_xta = _local_mi(samples, covariance, x, a, base=base)
    i_xtb = _local_mi(samples, covariance, x, b, base=base)
    i_yta = _local_mi(samples, covariance, y, a, base=base)
    i_ytb = _local_mi(samples, covariance, y, b, base=base)
    i_xyta = _local_mi(samples, covariance, x + y, a, base=base)
    i_xytb = _local_mi(samples, covariance, x + y, b, base=base)
    i_xtab = _local_mi(samples, covariance, x, a + b, base=base)
    i_ytab = _local_mi(samples, covariance, y, a + b, base=base)
    i_xytab = _local_mi(samples, covariance, x + y, a + b, base=base)

    r_xyta = _ccs_local(samples, covariance, x, y, a, base=base)
    r_xytb = _ccs_local(samples, covariance, x, y, b, base=base)
    r_xyta_b = _ccs_local(samples, covariance, x, y, a + b, base=base)
    r_abtx = _ccs_local(samples, covariance, a, b, x, base=base)
    r_abty = _ccs_local(samples, covariance, a, b, y, base=base)
    r_abtxy = _ccs_local(samples, covariance, a, b, x + y, base=base)

    double_coinfo = (
        -i_xta
        - i_xtb
        - i_yta
        - i_ytb
        + i_xtab
        + i_ytab
        + i_xyta
        + i_xytb
        - i_xytab
        + r_xyta
        + r_xytb
        - r_xyta_b
        + r_abtx
        + r_abty
        - r_abtxy
    )
    signs = torch.stack(
        [
            torch.sign(i_xta),
            torch.sign(i_xtb),
            torch.sign(i_yta),
            torch.sign(i_ytb),
            torch.sign(double_coinfo),
        ],
        dim=-1,
    )
    coherent = (signs == signs[..., :1]).all(-1)
    local = torch.where(coherent, double_coinfo, torch.zeros_like(double_coinfo))
    return local.mean(-1)


def _double_idep(
    covariance: torch.Tensor,
    sources,
    targets,
    *,
    redundancy: PhiIDRedundancy,
    base: float,
) -> torch.Tensor:
    """Directional closures for the single-target Gaussian I_dep prescription.

    The archived ComplexBox/Imperial PhiID implementations do not define public
    methods named ``Idep_a`` and ``Idep_b``. ComplexTorch therefore keeps the two
    possible PID-compatible orientations explicit: ``idep_a`` closes the free
    bottom node from forward PIDs, while ``idep_b`` uses the time-reversed PIDs.
    They must not be described as verified ELPH parity.
    """
    if redundancy == "idep_a":
        candidates = [
            _idep(covariance, sources[0], sources[1], target, base=base)
            for target in targets
        ]
    else:
        candidates = [
            _idep(covariance, targets[0], targets[1], source, base=base)
            for source in sources
        ]
    return torch.stack(candidates, dim=-1).amin(-1)


def _known_quantities(
    covariance: torch.Tensor,
    sources,
    targets,
    *,
    redundancy: PhiIDRedundancy,
    base: float,
    ccs_samples: torch.Tensor | None,
) -> torch.Tensor:
    x, y = sources
    a, b = targets

    i_xta = _mi(covariance, x, a, base=base)
    i_xtb = _mi(covariance, x, b, base=base)
    i_yta = _mi(covariance, y, a, base=base)
    i_ytb = _mi(covariance, y, b, base=base)
    i_xyta = _mi(covariance, x + y, a, base=base)
    i_xytb = _mi(covariance, x + y, b, base=base)
    i_xtab = _mi(covariance, x, a + b, base=base)
    i_ytab = _mi(covariance, y, a + b, base=base)
    i_xytab = _mi(covariance, x + y, a + b, base=base)

    r_xyta = _single_target_redundancy(
        covariance, x, y, a, redundancy=redundancy, base=base, ccs_samples=ccs_samples
    )
    r_xytb = _single_target_redundancy(
        covariance, x, y, b, redundancy=redundancy, base=base, ccs_samples=ccs_samples
    )
    r_xytab = _single_target_redundancy(
        covariance, x, y, a + b, redundancy=redundancy, base=base, ccs_samples=ccs_samples
    )
    r_abtx = _single_target_redundancy(
        covariance, a, b, x, redundancy=redundancy, base=base, ccs_samples=ccs_samples
    )
    r_abty = _single_target_redundancy(
        covariance, a, b, y, redundancy=redundancy, base=base, ccs_samples=ccs_samples
    )
    r_abtxy = _single_target_redundancy(
        covariance, a, b, x + y, redundancy=redundancy, base=base, ccs_samples=ccs_samples
    )

    if redundancy == "mmi":
        rtr = torch.stack([i_xta, i_xtb, i_yta, i_ytb], dim=-1).amin(-1)
    elif redundancy == "ccs":
        if ccs_samples is None:
            raise RuntimeError("CCS integration nodes were not initialized")
        rtr = _double_ccs(ccs_samples, covariance, sources, targets, base=base)
    else:
        rtr = _double_idep(
            covariance, sources, targets, redundancy=redundancy, base=base
        )

    return torch.stack(
        [
            rtr,
            r_xyta,
            r_xytb,
            r_xytab,
            r_abtx,
            r_abty,
            r_abtxy,
            i_xta,
            i_xtb,
            i_yta,
            i_ytb,
            i_xyta,
            i_xytb,
            i_xtab,
            i_ytab,
            i_xytab,
        ],
        dim=-1,
    )


def _atoms_from_knowns(knowns: torch.Tensor) -> torch.Tensor:
    matrix = torch.as_tensor(
        _KNOWNS_TO_ATOMS, dtype=knowns.dtype, device=knowns.device
    )
    batch_shape = knowns.shape[:-1]
    if batch_shape:
        matrix = matrix.expand(*batch_shape, -1, -1)
    return torch.linalg.solve(matrix, knowns.unsqueeze(-1)).squeeze(-1)


def gaussian_phiid_atoms(
    joint_covariance: torch.Tensor,
    block_size: int = 1,
    *,
    redundancy: PhiIDRedundancy | str = "mmi",
    base: float = 2.0,
    ccs_qmc_samples: int = 4096,
) -> dict[str, torch.Tensor]:
    """Compute the complete 16-atom Gaussian PhiID decomposition.

    Parameters
    ----------
    joint_covariance
        Joint covariance ordered ``[past0, past1, future0, future1]`` with four
        equally sized blocks. Leading batch dimensions are preserved.
    block_size
        Number of variables in each past/future block.
    redundancy
        ``"mmi"``, ``"ccs"``, ``"idep_a"``, or ``"idep_b"``.
    base
        Logarithm base used by all information quantities.
    ccs_qmc_samples
        Deterministic Sobol integration nodes used only by ``"ccs"``.

    Returns
    -------
    dict[str, torch.Tensor]
        The sixteen named atoms plus ``total`` and ``reconstruction``. Every
        output preserves the leading batch dimensions of ``joint_covariance``.

    Notes
    -----
    MMI and I_dep are analytic functions of covariance. Gaussian CCS is a
    deterministic numerical expectation under the supplied model distribution;
    the integration nodes are not observations and are never used to estimate a
    covariance.

    ``idep_a`` and ``idep_b`` are the two directional PhiID closures of the
    single-target Gaussian I_dep PID. The current archived ELPH/ComplexBox
    sources do not assign these names, so ComplexTorch documents them as
    directional closures rather than claiming external naming parity.

    Varley's :math:`I_{\\tau sx}` is deliberately rejected here because the
    published construction is discrete-only and has no established continuous
    Gaussian generalization.
    """
    covariance = _validate_covariance(joint_covariance, block_size)
    method = _validate_redundancy(str(redundancy))
    if base <= 0 or base == 1:
        raise ValueError("base must be positive and different from one")

    sources = (
        tuple(range(0, block_size)),
        tuple(range(block_size, 2 * block_size)),
    )
    targets = (
        tuple(range(2 * block_size, 3 * block_size)),
        tuple(range(3 * block_size, 4 * block_size)),
    )

    ccs_samples = None
    if method == "ccs":
        ccs_samples = _sobol_gaussian_samples(covariance, ccs_qmc_samples)
        _LOG.debug(
            "PhiID CCS: qmc_samples=%d block_size=%d base=%s dtype=%s device=%s",
            ccs_qmc_samples,
            block_size,
            base,
            covariance.dtype,
            covariance.device,
        )
    else:
        _LOG.debug(
            "PhiID %s: analytic covariance path block_size=%d base=%s dtype=%s device=%s",
            method,
            block_size,
            base,
            covariance.dtype,
            covariance.device,
        )

    knowns = _known_quantities(
        covariance,
        sources,
        targets,
        redundancy=method,
        base=base,
        ccs_samples=ccs_samples,
    )
    atoms = _atoms_from_knowns(knowns)
    result = {label: atoms[..., index] for index, label in enumerate(ATOM_LABELS)}
    result["total"] = knowns[..., -1]
    result["reconstruction"] = atoms.sum(-1)
    _LOG.debug(
        "PhiID %s: max reconstruction residual=%g",
        method,
        float((result["reconstruction"] - result["total"]).abs().amax().detach().cpu()),
    )
    return result


def gaussian_phiid_mmi(
    joint_covariance: torch.Tensor,
    n_past_x: int = 1,
    n_future_x: int = 1,
    *,
    base: float = 2.0,
) -> dict[str, torch.Tensor]:
    """Backward-compatible source-mode aggregates derived from MMI PhiID atoms."""
    if n_past_x != n_future_x:
        raise ValueError("equal block sizes are required")
    atoms = gaussian_phiid_atoms(
        joint_covariance,
        n_past_x,
        redundancy="mmi",
        base=base,
    )
    zero = torch.zeros_like(atoms["total"])
    return {
        "redundant": sum(
            (value for key, value in atoms.items() if key.startswith("red_to_")), zero
        ),
        "unique_x": sum(
            (value for key, value in atoms.items() if key.startswith("unq0_to_")), zero
        ),
        "unique_y": sum(
            (value for key, value in atoms.items() if key.startswith("unq1_to_")), zero
        ),
        "synergistic": sum(
            (value for key, value in atoms.items() if key.startswith("syn_to_")), zero
        ),
        "total": atoms["total"],
        "atoms": atoms,
    }


__all__ = [
    "ATOM_LABELS",
    "PhiIDRedundancy",
    "gaussian_phiid_atoms",
    "gaussian_phiid_mmi",
]
