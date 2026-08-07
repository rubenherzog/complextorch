"""Public model-order selection API for VAR and state-space models.

The package separates model families while sharing temporal-validation
infrastructure internally. ``selection_var`` contains VAR lag-order selectors;
``selection_state_space`` contains latent state-dimension selectors; and
``_temporal`` provides the private leakage-safe cross-validation engine used by
both. Model fitting remains in :mod:`complextorch.var` and
:mod:`complextorch.state_space`.
"""

from ._temporal import EpochTimeSeriesSplit, TemporalFold
from .selection_var import (
    VARInformationCriteriaResult,
    VAROrderScore,
    VAROrderSearchCV,
    VAROrderSearchResult,
    VAROrderSelectionIC,
)
from .selection_state_space import (
    StateSpaceOrderScore,
    StateSpaceOrderSearchCV,
    StateSpaceOrderSearchResult,
    StateSpaceOrderSelection,
    StateSpaceOrderSelectionResult,
)

__all__ = [
    "EpochTimeSeriesSplit",
    "TemporalFold",
    "VAROrderSelectionIC",
    "VARInformationCriteriaResult",
    "VAROrderSearchCV",
    "VAROrderSearchResult",
    "VAROrderScore",
    "StateSpaceOrderSelection",
    "StateSpaceOrderSelectionResult",
    "StateSpaceOrderSearchCV",
    "StateSpaceOrderSearchResult",
    "StateSpaceOrderScore",
]
