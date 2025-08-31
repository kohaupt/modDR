import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "moddr"
author = "Konstantin Haupt"
copyright = "2025, Konstantin Haupt"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

master_doc = "index"
source_suffix = [".rst", ".md"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
