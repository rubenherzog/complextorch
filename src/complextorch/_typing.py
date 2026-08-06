"""Shared public type aliases used throughout ComplexTorch.

The package accepts both :class:`numpy.ndarray` and :class:`torch.Tensor`
inputs at API boundaries. Numerical routines normalize these values to Torch
tensors before performing batched calculations.
"""
from __future__ import annotations

from typing import TypeAlias

import numpy as np
import torch

ArrayLike: TypeAlias = np.ndarray | torch.Tensor
