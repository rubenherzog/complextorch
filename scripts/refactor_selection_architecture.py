"""One-shot refactor of ComplexTorch model-selection architecture.

This script is intentionally removed by the workflow after the transformed
source tree passes the full validation matrix.
"""
from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "complextorch"
SELECTION = SRC / "selection"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def between(text: str, start: str, end: str | None = None) -> str:
    i = text.index(start)
    if end is None:
        return text[i:]
    j = text.index(end, i)
    return text[i:j]


selection_old = read(SRC / "selection.py")
var_cv_old = read(SRC / "var_selection_cv.py")
ss_cv_old = read(SRC / "state_space_selection.py")
temporal_old = read(SRC / "_temporal_order_search.py")
subspace_old = read(SRC / "_subspace.py")
model_comparison_old = read(SRC / "_model_comparison.py")

# ---------------------------------------------------------------------------
# Shared temporal selection infrastructure.
# ---------------------------------------------------------------------------
split_block = between(
    selection_old,
    "@dataclass(frozen=True)\nclass TemporalFold:",
    "\n\ndef _information_criteria(",
).rstrip()

temporal = temporal_old.replace("from ._typing import ArrayLike\n", "")
temporal = temporal.replace(
    "from .selection import EpochTimeSeriesSplit, TemporalFold\n", ""
)
temporal = temporal.replace("ArrayLike", "np.ndarray | torch.Tensor")
temporal = temporal.replace(
    "\n\nPredictionMode =",
    "\n\n" + split_block + "\n\n\nPredictionMode =",
    1,
)
write(SELECTION / "_temporal.py", temporal)

# ---------------------------------------------------------------------------
# VAR selection: IC and temporal CV have one implementation and source of truth.
# ---------------------------------------------------------------------------
information_criteria = between(
    selection_old,
    "def _information_criteria(",
    "\n\ndef _estimator_information_criteria(",
).rstrip()
var_ic = between(
    selection_old,
    "@dataclass(frozen=True)\nclass VARInformationCriteriaResult:",
    "\n@dataclass(frozen=True)\nclass StateSpaceOrderSelectionResult:",
).rstrip()
var_ic = var_ic.replace("ArrayLike", "np.ndarray | torch.Tensor")

selection_var = var_cv_old
selection_var = selection_var.replace(
    "from ._temporal_order_search import _TemporalOrderSearchCV",
    "from ._temporal import _TemporalOrderSearchCV, EpochTimeSeriesSplit",
)
selection_var = selection_var.replace("from .linalg import", "from ..linalg import")
selection_var = selection_var.replace(
    "from .selection import EpochTimeSeriesSplit, _information_criteria\n", ""
)
selection_var = selection_var.replace("from .var import VAR", "from ..var import VAR")
selection_var = selection_var.replace(
    "import torch\n", "import torch\nfrom sklearn.base import BaseEstimator\n", 1
)
selection_var = selection_var.replace(
    "\n\n@dataclass(frozen=True)\nclass VAROrderScore:",
    "\n\n" + information_criteria + "\n\n\n" + var_ic
    + "\n\n\n@dataclass(frozen=True)\nclass VAROrderScore:",
    1,
)
write(SELECTION / "selection_var.py", selection_var)

# ---------------------------------------------------------------------------
# State-space selection: Bauer/order-selection logic lives with the selectors;
# Larimore decomposition itself remains in the shared subspace math module.
# ---------------------------------------------------------------------------
normalise_selection = between(
    subspace_old,
    "def _normalise_observations(",
    "\n\ndef _bauer_svc(",
).rstrip()
bauer = between(
    subspace_old,
    "def _bauer_svc(",
    "\n\n@dataclass(frozen=True)\nclass _StateSpaceOrderComputation:",
).rstrip()
order_result = between(
    subspace_old,
    "@dataclass(frozen=True)\nclass _StateSpaceOrderComputation:",
    "\n\ndef _block_hankel(",
).rstrip()
order_compute = between(
    subspace_old,
    "def _larimore_state_space_order(",
    "\n\ndef _larimore_decomposition(",
).rstrip()
selection_helpers = "\n\n\n".join(
    [normalise_selection, bauer, order_result, order_compute]
).replace("ArrayLike", "np.ndarray | torch.Tensor")
selection_helpers = selection_helpers.replace(
    "correlations = _canonical_correlations(\n        past, future_blocks, ridge=ridge\n    )",
    "correlations, _, _ = _larimore_decomposition(\n        past, future_blocks, ridge=ridge\n    )",
)

state_selector = between(
    selection_old,
    "@dataclass(frozen=True)\nclass StateSpaceOrderSelectionResult:",
).rstrip().replace("ArrayLike", "np.ndarray | torch.Tensor")

selection_state = ss_cv_old
selection_state = selection_state.replace(
    "from ._state_space_order import _bauer_svc, _block_hankel",
    "from .._subspace import _block_hankel, _larimore_decomposition, _resolve_dtype",
)
selection_state = selection_state.replace(
    "from ._temporal_order_search import _TemporalOrderSearchCV",
    "from ._temporal import _TemporalOrderSearchCV, EpochTimeSeriesSplit",
)
selection_state = selection_state.replace("from .control import", "from ..control import")
selection_state = selection_state.replace("from .linalg import", "from ..linalg import")
selection_state = selection_state.replace(
    "from .selection import EpochTimeSeriesSplit\n", ""
)
selection_state = selection_state.replace("from .state_space import (", "from ..state_space import (")
selection_state = selection_state.replace("    _larimore_decomposition,\n", "")
selection_state = selection_state.replace(
    "import torch\n", "import torch\nfrom sklearn.base import BaseEstimator\n", 1
)
selection_state = selection_state.replace(
    "\n\n@dataclass(frozen=True)\nclass StateSpaceOrderScore:",
    "\n\n" + selection_helpers + "\n\n\n" + state_selector
    + "\n\n\n@dataclass(frozen=True)\nclass StateSpaceOrderScore:",
    1,
)
write(SELECTION / "selection_state_space.py", selection_state)

# ---------------------------------------------------------------------------
# Public selection namespace.
# ---------------------------------------------------------------------------
selection_init = '''"""Model-order selection for VAR and state-space models."""

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
'''
write(SELECTION / "__init__.py", selection_init)

# ---------------------------------------------------------------------------
# Shared subspace math: fixed-order Larimore/N4SID primitives only.
# ---------------------------------------------------------------------------
resolve_dtype = between(
    subspace_old, "def _resolve_dtype(", "\n\ndef _normalise_observations("
).rstrip()
block_hankel = between(
    subspace_old, "def _block_hankel(", "\n\ndef _canonical_correlations("
).rstrip()
larimore_decomposition = between(
    subspace_old, "def _larimore_decomposition("
).rstrip()
subspace_new = r'''"""Shared subspace-identification primitives.

This private module contains mathematical operations reused by fixed-order
state-space estimators and state-order selectors. It deliberately does not own
model-selection policy or public estimators.

Larimore CVA whitens past/future block-Hankel covariances and decomposes the
whitened cross-covariance. Selection criteria such as Bauer SVC live under
``complextorch.selection``.
"""
from __future__ import annotations

import torch

''' + resolve_dtype + "\n\n\n" + block_hankel + "\n\n\n" + larimore_decomposition
write(SRC / "_subspace.py", subspace_new)

# State-space estimators import their mathematical dependencies directly.
state_space_path = SRC / "state_space.py"
state_space = read(state_space_path).replace(
    "from ._state_space_order import _block_hankel, _resolve_dtype",
    "from ._subspace import _block_hankel, _larimore_decomposition, _resolve_dtype",
)
write(state_space_path, state_space)

# ---------------------------------------------------------------------------
# Nested VAR comparison is measure-specific, not model selection or VAR fitting.
# ---------------------------------------------------------------------------
nested_var = model_comparison_old.replace(
    "Shared nested-model primitives for empirical Gaussian comparisons.",
    "Nested-VAR primitives used by empirical predictive measures.",
)
nested_var = nested_var.replace("from .linalg import", "from ..linalg import")
nested_var = nested_var.replace("from .var import VAR", "from ..var import VAR")
nested_var = nested_var.replace(
    "from .measures.dynamics import cross_spectral_density",
    "from .dynamics import cross_spectral_density",
)
write(SRC / "measures" / "_nested_var.py", nested_var)

mvgc_path = SRC / "measures" / "mvgc.py"
mvgc = read(mvgc_path).replace(
    "from ._model_comparison import", "from ._nested_var import"
)
write(mvgc_path, mvgc)

# ---------------------------------------------------------------------------
# Package root: one public selection namespace, no sys.modules aliases or
# runtime monkey-patching.
# ---------------------------------------------------------------------------
init_path = SRC / "__init__.py"
init_lines = read(init_path).splitlines()
filtered = []
for line in init_lines:
    stripped = line.strip()
    if stripped in {
        "import sys as _sys",
        "from . import _subspace as _subspace",
        "_sys.modules[__name__ + \"._state_space_order\"] = _subspace",
        "from . import state_space as _state_space",
        "_state_space._larimore_decomposition = _subspace._larimore_decomposition",
    }:
        continue
    if line.startswith("from .selection import "):
        continue
    if line.startswith("from .var_selection_cv import "):
        continue
    if line.startswith("from .state_space_selection import "):
        continue
    if stripped.startswith("# Temporary private import alias"):
        continue
    if stripped.startswith("# Preserve ``complextorch.selection``"):
        continue
    if stripped.startswith("# through the shared temporal-search implementation"):
        continue
    if line.startswith("from . import selection as _selection"):
        continue
    if line.startswith("_selection.VAROrder"):
        continue
    filtered.append(line)

selection_import = (
    "from .selection import EpochTimeSeriesSplit, VAROrderSelectionIC, "
    "VARInformationCriteriaResult, VAROrderSearchCV, VAROrderSearchResult, "
    "VAROrderScore, StateSpaceOrderSelection, StateSpaceOrderSelectionResult, "
    "StateSpaceOrderSearchCV, StateSpaceOrderSearchResult, StateSpaceOrderScore"
)
insert_at = next(
    i for i, line in enumerate(filtered)
    if line.startswith("from .representations import ")
) + 1
filtered.insert(insert_at, selection_import)
write(init_path, "\n".join(filtered))

# ---------------------------------------------------------------------------
# Remove stale paths and update any private test/example imports.
# ---------------------------------------------------------------------------
for path in (
    SRC / "selection.py",
    SRC / "var_selection_cv.py",
    SRC / "state_space_selection.py",
    SRC / "_temporal_order_search.py",
    SRC / "_typing.py",
    SRC / "_model_comparison.py",
):
    path.unlink()

for root in (SRC, ROOT / "tests", ROOT / "examples"):
    if not root.exists():
        continue
    for path in root.rglob("*.py"):
        text = read(path)
        text = text.replace(
            "complextorch.var_selection_cv", "complextorch.selection.selection_var"
        )
        text = text.replace(
            "complextorch.state_space_selection",
            "complextorch.selection.selection_state_space",
        )
        text = text.replace(
            "complextorch._temporal_order_search", "complextorch.selection._temporal"
        )
        text = text.replace(
            "complextorch._state_space_order",
            "complextorch.selection.selection_state_space",
        )
        text = text.replace(
            "from ._temporal_order_search import",
            "from .selection._temporal import",
        )
        text = text.replace(
            "from .var_selection_cv import",
            "from .selection.selection_var import",
        )
        text = text.replace(
            "from .state_space_selection import",
            "from .selection.selection_state_space import",
        )
        text = text.replace(
            "from ._state_space_order import",
            "from .selection.selection_state_space import",
        )
        if "_typing import ArrayLike" in text:
            text = re.sub(
                r"^from \.{1,2}_typing import ArrayLike\n", "", text, flags=re.M
            )
            text = text.replace("ArrayLike", "np.ndarray | torch.Tensor")
            if "np.ndarray" in text and "import numpy as np" not in text:
                marker = "from __future__ import annotations\n"
                text = text.replace(marker, marker + "\nimport numpy as np\n", 1)
            if "torch.Tensor" in text and "import torch" not in text:
                marker = "from __future__ import annotations\n"
                text = text.replace(marker, marker + "\nimport torch\n", 1)
        write(path, text)

# ---------------------------------------------------------------------------
# Architectural invariants: fail before tests if the refactor is incomplete.
# ---------------------------------------------------------------------------
forbidden_paths = [
    "selection.py",
    "var_selection_cv.py",
    "state_space_selection.py",
    "_temporal_order_search.py",
    "_typing.py",
    "_model_comparison.py",
    "_state_space_order.py",
]
for name in forbidden_paths:
    if (SRC / name).exists():
        raise RuntimeError(f"stale architecture file remains: {name}")

source_text = "\n".join(read(path) for path in SRC.rglob("*.py"))
for token in (
    "._state_space_order",
    "._temporal_order_search",
    ".var_selection_cv",
    ".state_space_selection",
    "._typing",
    "._model_comparison",
    "_sys.modules",
    "_selection.VAROrder",
):
    if token in source_text:
        raise RuntimeError(f"stale architecture reference remains: {token}")

for class_name in (
    "VAROrderSearchCV",
    "VAROrderSelectionIC",
    "StateSpaceOrderSearchCV",
    "StateSpaceOrderSelection",
):
    count = source_text.count(f"class {class_name}")
    if count != 1:
        raise RuntimeError(f"expected one {class_name} definition, found {count}")

if "class _TemporalOrderSearchCV" not in read(SELECTION / "_temporal.py"):
    raise RuntimeError("shared temporal CV engine missing")
if "def _bauer_svc" not in read(SELECTION / "selection_state_space.py"):
    raise RuntimeError("Bauer selection primitive not colocated with state-space selection")
if "def _bauer_svc" in read(SRC / "_subspace.py"):
    raise RuntimeError("selection policy leaked into _subspace.py")
if "fit_nested_var_models" not in read(SRC / "measures" / "_nested_var.py"):
    raise RuntimeError("nested VAR measure helpers were not moved into measures")

print("Selection architecture refactor completed and invariants passed.")
