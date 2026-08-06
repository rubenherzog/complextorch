r"""Backward-compatible imports for private state-space order primitives.

The implementation lives in :mod:`complextorch._subspace`. This private shim
is retained temporarily so internal imports and downstream development code do
not break during the architectural cleanup.
"""

from ._subspace import (
    _StateSpaceOrderComputation,
    _bauer_svc,
    _block_hankel,
    _canonical_correlations,
    _larimore_decomposition,
    _larimore_state_space_order,
    _normalise_observations,
    _resolve_dtype,
)

__all__ = [
    "_StateSpaceOrderComputation",
    "_bauer_svc",
    "_block_hankel",
    "_canonical_correlations",
    "_larimore_decomposition",
    "_larimore_state_space_order",
    "_normalise_observations",
    "_resolve_dtype",
]
