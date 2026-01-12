import os
import sys

# Add repository src root to Python path for autodoc
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = "mRNNTorch"
author = ""

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
]

# Mock heavy deps so autodoc works on Read the Docs
autodoc_mock_imports = [
    "torch",
    "numpy",
    "matplotlib",
    "sklearn",
    "tqdm",
]

templates_path = ["_templates"]
exclude_patterns = []

# MyST (Markdown) configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_title = "mRNNTorch"
html_static_path = ["_static"]
html_logo = "_static/logo.svg"
html_favicon = "_static/favicon.svg"
html_css_files = [
    "css/custom.css",
]

# -- Autodoc options ---------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": True,
    "show-inheritance": True,
}
