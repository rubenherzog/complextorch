"""Structural checks for API documentation intended for Sphinx autodoc."""
from __future__ import annotations

import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "src" / "complextorch"
PLACEHOLDERS = (
    "Input controlling ``",
    "Result described by the function name",
    "The class follows the scikit-learn fitted-attribute convention",
)
SCIENTIFIC_MODULES = {
    "confidence.py",
    "control.py",
    "inference.py",
    "inference_registry.py",
    "linalg.py",
    "representations.py",
    "selection.py",
    "simulate.py",
    "state_space.py",
    "var.py",
    "backbone.py",
    "cmem.py",
    "criticality.py",
    "discrete.py",
    "dynamics.py",
    "emergence.py",
    "gaussian.py",
    "hop.py",
    "mvgc.py",
    "mvgc_api.py",
    "oir.py",
    "pdgc.py",
    "phid.py",
    "pird.py",
    "primary.py",
    "secondary.py",
}


def _symbols(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def test_every_module_class_and_function_has_a_docstring():
    missing: list[str] = []
    for path in sorted(SOURCE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if not ast.get_docstring(tree):
            missing.append(f"{path.relative_to(SOURCE)}:<module>")
        for node in _symbols(tree):
            if not ast.get_docstring(node):
                missing.append(f"{path.relative_to(SOURCE)}:{node.lineno}:{node.name}")
    assert not missing, "Missing docstrings:\n" + "\n".join(missing)


def test_generated_docstrings_contain_no_placeholder_language():
    occurrences: list[str] = []
    for path in sorted(SOURCE.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for marker in PLACEHOLDERS:
            if marker in text:
                occurrences.append(f"{path.relative_to(SOURCE)}: {marker}")
    assert not occurrences, "Placeholder documentation remains:\n" + "\n".join(occurrences)


def test_scientific_modules_include_primary_references():
    missing: list[str] = []
    for path in sorted(SOURCE.rglob("*.py")):
        if path.name not in SCIENTIFIC_MODULES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        doc = ast.get_docstring(tree) or ""
        if "References" not in doc:
            missing.append(str(path.relative_to(SOURCE)))
    assert not missing, "Scientific modules without references:\n" + "\n".join(missing)
