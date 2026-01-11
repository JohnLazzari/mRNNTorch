"""Pytest configuration helpers for mRNNTorch tests.

This file keeps imports and global test setup minimal while ensuring that
the test suite can import the local package without installation and that
device defaults are safe for CPU-only test environments.
"""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
# Ensure the local src/ tree is importable without an installed package.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from mrnntorch.Region import region_base
except Exception:
    region_base = None

# If defaults omit a device, force CPU to avoid CUDA-only defaults in tests.
if region_base is not None and "device" not in region_base.DEFAULT_REC_REGIONS:
    region_base.DEFAULT_REC_REGIONS["device"] = "cpu"
