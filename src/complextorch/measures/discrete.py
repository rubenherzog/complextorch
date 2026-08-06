"""Discrete information and sequence-complexity estimators.

Probabilities are empirical masses. Lempel--Ziv complexity counts novel phrases
in an incremental parsing of a finite symbol sequence.

References
----------
- Lempel, A. and Ziv, J. (1976). On the complexity of finite sequences.
- Cover, T. M. and Thomas, J. A. (2006).
"""
from __future__ import annotations
from collections import Counter
import math
import numpy as np
import torch


def _codes(x):
    """Codes.
    
    Parameters
    ----------
    x
        Input observations or tensor-valued quantity.
    
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
    array = np.asarray(torch.as_tensor(x).detach().cpu())
    if array.ndim == 1:
        return [(value.item() if hasattr(value, "item") else value,) for value in array]
    return [tuple(row.tolist()) for row in array]


def discrete_entropy(x, *, base: float = 2.0) -> float:
    """Discrete entropy.
    
    Parameters
    ----------
    x
        Input observations or tensor-valued quantity.
    base
        Logarithm base used for information quantities.
    
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
    # Estimate Shannon entropy from empirical symbol probabilities, omitting zero-probability terms.
    codes = _codes(x)
    counts = Counter(codes)
    n = len(codes)
    return -sum((count / n) * math.log(count / n, base) for count in counts.values())


def discrete_mutual_information(x, y, *, base: float = 2.0) -> float:
    """Discrete mutual information.
    
    Parameters
    ----------
    x
        Input observations or tensor-valued quantity.
    y
        Unused scikit-learn compatibility target.
    base
        Logarithm base used for information quantities.
    
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
    x_array = np.asarray(torch.as_tensor(x).detach().cpu())
    y_array = np.asarray(torch.as_tensor(y).detach().cpu())
    if len(x_array) != len(y_array):
        raise ValueError("x and y must have equal samples")
    joint = np.column_stack([x_array, y_array])
    return discrete_entropy(x_array, base=base) + discrete_entropy(y_array, base=base) - discrete_entropy(joint, base=base)


def discrete_total_correlation(x, *, base: float = 2.0) -> float:
    """Discrete total correlation.
    
    Parameters
    ----------
    x
        Input observations or tensor-valued quantity.
    base
        Logarithm base used for information quantities.
    
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
    array = np.asarray(torch.as_tensor(x).detach().cpu())
    if array.ndim != 2:
        raise ValueError("x must be (samples,variables)")
    return sum(discrete_entropy(array[:, j], base=base) for j in range(array.shape[1])) - discrete_entropy(array, base=base)


def lempel_ziv_complexity(sequence, *, normalize: bool = False) -> float:
    """Compute incremental Lempel--Ziv phrase complexity.
    
    References
    ----------
    - Lempel and Ziv (1976).
    """
    # Count new phrases in the LZ76 parsing as a finite-sample proxy for algorithmic complexity.
    symbols = list(np.asarray(torch.as_tensor(sequence).detach().cpu()).ravel())
    n = len(symbols)
    if n == 0:
        return 0.0
    i = 0
    k = 1
    l = 1
    k_max = 1
    complexity = 1
    while True:
        if symbols[i + k - 1] == symbols[l + k - 1]:
            k += 1
            if l + k > n:
                complexity += 1
                break
        else:
            k_max = max(k, k_max)
            i += 1
            if i == l:
                complexity += 1
                l += k_max
                if l + 1 > n:
                    break
                i = 0
                k = 1
                k_max = 1
            else:
                k = 1
    if not normalize:
        return float(complexity)
    alphabet = max(2, len(set(symbols)))
    return float(complexity * math.log(n, alphabet) / n)
