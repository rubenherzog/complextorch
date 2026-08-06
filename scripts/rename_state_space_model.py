from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = "LinearDynamicalSystem"
NEW = "StateSpaceModel"
TEXT_SUFFIXES = {".py", ".md", ".rst", ".toml", ".txt", ".yml", ".yaml"}
SKIP_PARTS = {".git", ".venv", "dist", "build", "__pycache__"}

changed = []
for path in ROOT.rglob("*"):
    if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
        continue
    if any(part in SKIP_PARTS for part in path.parts):
        continue
    text = path.read_text(encoding="utf-8")
    if OLD not in text:
        continue
    path.write_text(text.replace(OLD, NEW), encoding="utf-8")
    changed.append(str(path.relative_to(ROOT)))

# The rename is intended to be complete: no compatibility alias remains in the
# public API or internal source. A stale occurrence therefore fails the job.
stale = []
for path in ROOT.rglob("*"):
    if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
        continue
    if any(part in SKIP_PARTS for part in path.parts):
        continue
    if OLD in path.read_text(encoding="utf-8"):
        stale.append(str(path.relative_to(ROOT)))
if stale:
    raise RuntimeError(f"stale {OLD} references remain: {stale}")

print("Renamed in:")
print("\n".join(changed))
