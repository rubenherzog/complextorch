"""Public model-order selection API for VAR and state-space models.

The package separates model families while sharing temporal-validation
infrastructure internally. ``selection_var`` contains VAR lag-order selectors;
``selection_state_space`` contains latent state-dimension selectors; and
``_temporal`` provides the private leakage-safe cross-validation engine used by
both. Model fitting remains in :mod:`complextorch.var` and
:mod:`complextorch.state_space`.
"""

from ._temporal import EpochTimeSeriesSplit, TemporalFold
from .selection_state_space import (
    StateSpaceOrderScore,
    StateSpaceOrderSearchResult,
    StateSpaceOrderSelection,
    StateSpaceOrderSelectionResult,
)
from .selection_state_space_cv import StateSpaceOrderSearchCV
from .selection_var import (
    VARInformationCriteriaResult,
    VAROrderScore,
    VAROrderSearchCV,
    VAROrderSearchResult,
    VAROrderSelectionIC,
)

__all__ = [
    "EpochTimeSeriesSplit",
    "StateSpaceOrderScore",
    "StateSpaceOrderSearchCV",
    "StateSpaceOrderSearchResult",
    "StateSpaceOrderSelection",
    "StateSpaceOrderSelectionResult",
    "TemporalFold",
    "VARInformationCriteriaResult",
    "VAROrderScore",
    "VAROrderSearchCV",
    "VAROrderSearchResult",
    "VAROrderSelectionIC",
]
