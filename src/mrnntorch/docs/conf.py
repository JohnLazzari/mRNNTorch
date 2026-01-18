import os
import sys
import types

# Add repository src root to Python path for autodoc
_CONF_DIR = os.path.abspath(os.path.dirname(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_CONF_DIR, "../../.."))
sys.path.insert(0, _SRC_DIR)

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

# Provide a minimal torch stub for autodoc when torch isn't installed.
try:
    import torch  # noqa: F401
except Exception:  # pragma: no cover - docs-only fallback
    torch_stub = types.ModuleType("torch")
    torch_nn_stub = types.ModuleType("torch.nn")
    torch_nn_functional_stub = types.ModuleType("torch.nn.functional")

    class _TorchModule:  # minimal base class for nn.Module inheritance
        pass

    class _TorchTensor:
        pass

    torch_stub.Tensor = _TorchTensor
    torch_stub.float32 = "float32"
    torch_stub.complex64 = "complex64"
    torch_stub.nn = torch_nn_stub
    torch_nn_stub.Module = _TorchModule
    torch_nn_stub.functional = torch_nn_functional_stub

    sys.modules["torch"] = torch_stub
    sys.modules["torch.nn"] = torch_nn_stub
    sys.modules["torch.nn.functional"] = torch_nn_functional_stub

# Mock heavy deps so autodoc works on Read the Docs
autodoc_mock_imports = [
    "numpy",
    "matplotlib",
    "sklearn",
    "sklearn.decomposition",
    "tqdm",
    "tqdm.auto",
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
