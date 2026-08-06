"""Gaussian MMI PhiID on the complete 4x4 double-redundancy lattice.

Notes
-----
Gaussian PhiID decomposes time-delayed mutual information into integrated
information atoms. The implementation uses the minimum-mutual-information
redundancy prescription.

References
----------
- Mediano, P. A. M. et al. (2021). Towards an extended taxonomy of information
  dynamics via integrated information decomposition.
- dit PhiID-related implementations: https://github.com/Imperial-MIND-lab/integrated-info-decomp
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
    """ leq.
    
    Parameters
    ----------
    alpha
        Input controlling ``_leq``.
    beta
        Input controlling ``_leq``.
    
    Returns
    -------
    object
        Result described by the function name and annotated return type.
    
    Notes
    -----
    Tensor batch dimensions are preserved unless the public API explicitly
    documents a squeeze operation. Numerical validation is performed by the
    module before the core calculation.
    """
    return all(any(a.issubset(b) for a in alpha) for b in beta)


def _subcov(covariance, indices):
    """ subcov.
    
    Parameters
    ----------
    covariance
        Input controlling ``_subcov``.
    indices
        Input controlling ``_subcov``.
    
    Returns
    -------
    object
        Result described by the function name and annotated return type.
    
    Notes
    -----
    Tensor batch dimensions are preserved unless the public API explicitly
    documents a squeeze operation. Numerical validation is performed by the
    module before the core calculation.
    """
    index = torch.as_tensor(indices, dtype=torch.long, device=covariance.device)
    return covariance.index_select(-2, index).index_select(-1, index)


def _mi(covariance, left, right):
    """ mi.
    
    Parameters
    ----------
    covariance
        Input controlling ``_mi``.
    left
        Input controlling ``_mi``.
    right
        Input controlling ``_mi``.
    
    Returns
    -------
    object
        Result described by the function name and annotated return type.
    
    Notes
    -----
    Tensor batch dimensions are preserved unless the public API explicitly
    documents a squeeze operation. Numerical validation is performed by the
    module before the core calculation.
    """
    block = _subcov(covariance, tuple(left) + tuple(right))
    return gaussian_mutual_information(block, len(left))


def gaussian_phiid_atoms(joint_covariance: torch.Tensor, block_size: int = 1) -> dict[str, torch.Tensor]:
    """Return all 16 MMI PhiID atoms for [past0,past1,future0,future1].
            
            Compute Gaussian PhiID atoms under MMI redundancy.
            
            References
            ----------
            Mediano et al. (2021), integrated information decomposition.
        
        Compute Gaussian PhiID atoms under MMI redundancy.
        
        References
        ----------
        Mediano et al. (2021), integrated information decomposition.
    
    Compute Gaussian PhiID atoms under MMI redundancy.
    
    References
    ----------
    Mediano et al. (2021), integrated information decomposition.
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
