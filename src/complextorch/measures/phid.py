"""Gaussian integrated information decomposition (PhiID).

Time-delayed mutual information is decomposed into integrated information atoms
using the minimum-mutual-information redundancy prescription.

References
----------
- Mediano, P. A. M. et al. (2021). Integrated information decomposition.
- Reference implementation: https://github.com/Imperial-MIND-lab/integrated-info-decomp
"""
from __future__ import annotations
from itertools import product
import torch
from .gaussian import gaussian_mutual_information

ANTICHAINS = (
    frozenset({frozenset({0}), frozenset({1})}),
    frozenset({frozenset({0})}),
    frozenset({frozenset({1})}),
    frozenset({frozenset({0, 1})}),
)
LABELS = ("red", "unq0", "unq1", "syn")


def _leq(alpha, beta):
    """Leq.
    
    Parameters
    ----------
    alpha
        Non-negative ridge regularization strength.
    beta
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
    return all(any(a.issubset(b) for a in alpha) for b in beta)


def _subcov(covariance, indices):
    """Subcov.
    
    Parameters
    ----------
    covariance
        Symmetric covariance matrix or batch of covariance matrices.
    indices
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
    index = torch.as_tensor(indices, dtype=torch.long, device=covariance.device)
    return covariance.index_select(-2, index).index_select(-1, index)


def _mi(covariance, left, right):
    """Mi.
    
    Parameters
    ----------
    covariance
        Symmetric covariance matrix or batch of covariance matrices.
    left
        Input required by this calculation.
    right
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
    block = _subcov(covariance, tuple(left) + tuple(right))
    return gaussian_mutual_information(block, len(left))


def gaussian_phiid_atoms(joint_covariance: torch.Tensor, block_size: int = 1) -> dict[str, torch.Tensor]:
    """Compute Gaussian PhiID atoms under MMI redundancy.
    
    References
    ----------
    - Mediano et al. (2021).
    """
    covariance = torch.as_tensor(joint_covariance)
    if covariance.shape[-1] != 4 * block_size:
        raise ValueError("covariance dimension must equal 4 * block_size")
    sources = (tuple(range(0, block_size)), tuple(range(block_size, 2 * block_size)))
    targets = (tuple(range(2 * block_size, 3 * block_size)), tuple(range(3 * block_size, 4 * block_size)))
    nodes = list(product(range(4), range(4)))
    redundancies = {}
    for i, j in nodes:
        values = []
        for source_antichain in ANTICHAINS[i]:
            left = sum((sources[x] for x in sorted(source_antichain)), ())
            for target_antichain in ANTICHAINS[j]:
                right = sum((targets[x] for x in sorted(target_antichain)), ())
                values.append(_mi(covariance, left, right))
        redundancies[(i, j)] = torch.stack(values).amin(0)
    atoms = {}
    for node in nodes:
        lower = [other for other in nodes if other != node and _leq(ANTICHAINS[other[0]], ANTICHAINS[node[0]]) and _leq(ANTICHAINS[other[1]], ANTICHAINS[node[1]])]
        atoms[node] = redundancies[node] - sum((atoms[other] for other in lower), torch.zeros_like(redundancies[node]))
    result = {f"{LABELS[i]}_to_{LABELS[j]}": atoms[(i, j)] for i, j in nodes}
    result["total"] = _mi(covariance, sources[0] + sources[1], targets[0] + targets[1])
    result["reconstruction"] = torch.stack([atoms[node] for node in nodes]).sum(0)
    return result


def gaussian_phiid_mmi(joint_covariance: torch.Tensor, n_past_x: int = 1, n_future_x: int = 1) -> dict[str, torch.Tensor]:
    """Backward-compatible aggregates derived from the complete atom table."""
    if n_past_x != n_future_x:
        raise ValueError("equal block sizes are required")
    atoms = gaussian_phiid_atoms(joint_covariance, n_past_x)
    zero = torch.zeros_like(atoms["total"])
    redundant = sum((value for key, value in atoms.items() if key.startswith("red_to_")), zero)
    unique_x = sum((value for key, value in atoms.items() if key.startswith("unq0_to_")), zero)
    unique_y = sum((value for key, value in atoms.items() if key.startswith("unq1_to_")), zero)
    synergy = sum((value for key, value in atoms.items() if key.startswith("syn_to_")), zero)
    return {"redundant": redundant, "unique_x": unique_x, "unique_y": unique_y, "synergistic": synergy, "total": atoms["total"], "atoms": atoms}
