"""Sphinx directive for compact, human-readable API index tables."""

from __future__ import annotations

import importlib
import inspect
import os
from pathlib import Path
from typing import Any

from docutils import nodes
from sphinx import addnodes
from sphinx.util.docutils import SphinxDirective


def _resolve_object(dotted_name: str) -> Any:
    parts = dotted_name.split(".")
    for index in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:index])
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        obj: Any = module
        for attr in parts[index:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(f"Could not resolve {dotted_name!r}")


def _summary(obj: Any) -> str:
    doc = inspect.getdoc(obj) or ""
    for line in doc.splitlines():
        text = line.strip()
        if text:
            return text
    return "Public ComplexTorch API object."


def _source_location(obj: Any) -> tuple[Path | None, int | None]:
    try:
        source_file = inspect.getsourcefile(obj)
        if source_file is None:
            return None, None
        _, line = inspect.getsourcelines(obj)
        return Path(source_file).resolve(), line
    except (OSError, TypeError):
        module = inspect.getmodule(obj)
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            return None, None
        return Path(module_file).resolve(), None


def _github_source_url(obj: Any) -> str:
    source_file, line = _source_location(obj)
    ref = (
        os.environ.get("READTHEDOCS_GIT_COMMIT_HASH")
        or os.environ.get("GITHUB_SHA")
        or "main"
    )
    if source_file is None:
        return f"https://github.com/rubenherzog/complextorch/tree/{ref}/src/complextorch"

    try:
        marker = source_file.parts.index("src")
        relative = Path(*source_file.parts[marker:]).as_posix()
    except ValueError:
        relative = source_file.name

    url = f"https://github.com/rubenherzog/complextorch/blob/{ref}/{relative}"
    if line is not None:
        url += f"#L{line}"
    return url


def _entry(content: nodes.Node) -> nodes.entry:
    entry = nodes.entry()
    entry += content
    return entry


class ApiTableDirective(SphinxDirective):
    """Render a four-column API table from fully qualified Python objects."""

    has_content = True

    def run(self) -> list[nodes.Node]:
        names = [line.strip() for line in self.content if line.strip()]
        table = nodes.table(classes=["api-summary-table"])
        tgroup = nodes.tgroup(cols=4)
        table += tgroup
        for width in (24, 58, 9, 9):
            tgroup += nodes.colspec(colwidth=width)

        thead = nodes.thead()
        tgroup += thead
        header = nodes.row()
        for label in ("Name", "Description", "Docs", "Source"):
            paragraph = nodes.paragraph()
            paragraph += nodes.strong(text=label)
            header += _entry(paragraph)
        thead += header

        tbody = nodes.tbody()
        tgroup += tbody

        for dotted_name in names:
            obj = _resolve_object(dotted_name)
            row = nodes.row()

            name_paragraph = nodes.paragraph()
            name_paragraph += nodes.literal(text=dotted_name.rsplit(".", 1)[-1])
            row += _entry(name_paragraph)

            description = nodes.paragraph(text=_summary(obj))
            row += _entry(description)

            docs_paragraph = nodes.paragraph()
            docs_link = addnodes.pending_xref(
                "",
                refdomain="std",
                reftype="doc",
                reftarget=f"generated/{dotted_name}",
                refexplicit=True,
            )
            docs_link += nodes.inline(text="[docs]")
            docs_paragraph += docs_link
            row += _entry(docs_paragraph)

            source_paragraph = nodes.paragraph()
            source_paragraph += nodes.reference(
                "", "[source]", refuri=_github_source_url(obj)
            )
            row += _entry(source_paragraph)

            tbody += row

        return [table]


def setup(app: Any) -> dict[str, Any]:
    app.add_directive("api-table", ApiTableDirective)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
