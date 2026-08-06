"""Repair mechanical renames and validate unified order-selection exports."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "complextorch"

internal = SRC / "_state_space_order.py"
text = internal.read_text(encoding="utf-8")
text = text.replace("__bauer_svc", "_bauer_svc")
internal.write_text(text, encoding="utf-8")

selection = (SRC / "selection.py").read_text(encoding="utf-8")
required = (
    "class StateSpaceOrderSelectionResult:",
    "class StateSpaceOrderSelection(BaseEstimator):",
    "from ._state_space_order import _larimore_state_space_order",
)
for marker in required:
    if marker not in selection:
        raise RuntimeError(f"missing selection API marker: {marker}")

init = (SRC / "__init__.py").read_text(encoding="utf-8")
for legacy in (
    "LarimoreStateSpaceOrder",
    "LarimoreStateSpaceOrderResult",
    "bauer_svc",
    "larimore_state_space_order",
):
    if legacy in init:
        raise RuntimeError(f"legacy top-level export remains: {legacy}")
for public in ("StateSpaceOrderSelection", "StateSpaceOrderSelectionResult"):
    if public not in init:
        raise RuntimeError(f"missing top-level selection export: {public}")

# Compile every touched Python source before committing.
for path in (
    SRC / "selection.py",
    SRC / "_state_space_order.py",
    SRC / "__init__.py",
    ROOT / "tests" / "test_state_space_model_order.py",
    ROOT / "examples" / "state_space_order_selection.py",
):
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
