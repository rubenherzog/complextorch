"""Experimental Gaussian MMI PhiID aggregates.

The current implementation returns four predictive-information modes whose
sum equals the total mutual information between a bivariate past and future.
It does not claim the full 16-atom double-redundancy lattice yet.
"""
from __future__ import annotations
import torch
from .gaussian import gaussian_mutual_information


def gaussian_phiid_mmi(joint_covariance: torch.Tensor) -> dict[str, torch.Tensor]:
    """MMI aggregate decomposition for [past_x,past_y,future_x,future_y]."""
    covariance = torch.as_tensor(joint_covariance)
    dimension = covariance.shape[-1]
    if dimension % 4 != 0:
        raise ValueError("joint covariance must contain four equal blocks")
    block = dimension // 4
    past_x = list(range(0, block))
    past_y = list(range(block, 2 * block))
    future_x = list(range(2 * block, 3 * block))
    future_y = list(range(3 * block, 4 * block))

    def subset(indices):
        index = torch.tensor(indices, device=covariance.device)
        return covariance.index_select(-2, index).index_select(-1, index)

    def mutual_information(left, right):
        return gaussian_mutual_information(subset(left + right), len(left))

    total = mutual_information(past_x + past_y, future_x + future_y)
    information_x = mutual_information(past_x, future_x + future_y)
    information_y = mutual_information(past_y, future_x + future_y)
    redundant = torch.minimum(information_x, information_y)
    unique_x = (information_x - redundant).clamp_min(0)
    unique_y = (information_y - redundant).clamp_min(0)
    synergistic = (total - redundant - unique_x - unique_y).clamp_min(0)
    return {
        "redundant": redundant,
        "unique_x": unique_x,
        "unique_y": unique_y,
        "synergistic": synergistic,
        "total": total,
    }
