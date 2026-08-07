"""Sphinx configuration for the ComplexTorch documentation."""

from __future__ import annotations

import complextorch

project = "ComplexTorch"
author = "Rubén Herzog"
release = complextorch.__version__
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
]

autosummary_generate = True
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"/plot_",
    "ignore_pattern": r"state_space_order_selection\.py",
    "abort_on_example_error": True,
    "download_all_examples": False,
    "remove_config_comments": True,
    "show_memory": False,
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_static_path: list[str] = []
