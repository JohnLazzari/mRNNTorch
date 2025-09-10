import os
import sys
import types

# Add repository root to Python path for autodoc
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "mRNNTorch"
author = ""

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
]

# Mock heavy deps so autodoc works on Read the Docs
autodoc_mock_imports = [
    "torch",
    "numpy",
    "matplotlib",
    "sklearn",
]

# Stub package path used in code to avoid import errors during autodoc
# The repository currently keeps modules at the root, while code imports
# from "mRNNTorch.Region". Create a minimal stub to satisfy the import.
m_pkg = types.ModuleType("mRNNTorch")
m_region = types.ModuleType("mRNNTorch.Region")

class _Stub:  # minimal placeholders for class names used at import time
    pass

m_region.RecurrentRegion = _Stub
m_region.InputRegion = _Stub

sys.modules.setdefault("mRNNTorch", m_pkg)
sys.modules.setdefault("mRNNTorch.Region", m_region)

templates_path = ["_templates"]
exclude_patterns = []

# MyST (Markdown) configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Autodoc options ---------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": True,
    "show-inheritance": True,
}
