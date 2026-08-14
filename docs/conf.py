"""Sphinx configuration for the ComplexTorch documentation."""

from __future__ import annotations

import sys
from pathlib import Path

import complextorch

sys.path.insert(0, str(Path(__file__).parent / "_ext"))

project = "ComplexTorch"
author = "Rubén Herzog"
release = complextorch.__version__
version = release

extensions = [
    "api_table",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
    "sphinx_wagtail_theme",
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

html_theme = "sphinx_wagtail_theme"
html_theme_options = {
    "project_name": "ComplexTorch",
    "github_url": "https://github.com/rubenherzog/complextorch/blob/main/docs/",
    "footer_links": "Maintained by <a href=\"https://github.com/rubenherzog\">Rubén Herzog</a>",
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
