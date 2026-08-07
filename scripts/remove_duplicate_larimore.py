"""Temporary refactor helper: remove the duplicate Larimore decomposition."""
from pathlib import Path

path = Path("src/complextorch/state_space.py")
text = path.read_text(encoding="utf-8")
start = text.find("\ndef _larimore_decomposition(")
if start == -1:
    raise RuntimeError("state_space.py does not contain the expected duplicate _larimore_decomposition")
end = text.find("\ndef _fit_innovations_state_space_from_states(", start)
if end == -1:
    raise RuntimeError("could not locate the function following _larimore_decomposition")
text = text[:start] + "\n" + text[end:]
path.write_text(text, encoding="utf-8")

remaining = text.count("def _larimore_decomposition(")
if remaining:
    raise RuntimeError("state_space.py still defines _larimore_decomposition")
