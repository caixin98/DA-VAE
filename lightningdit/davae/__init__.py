"""
LightningDiT.vae
================

Expose DA-VAE LDM modules as a proper Python package so downstream modules
can import them without mutating ``sys.path`` at runtime. We also keep a
compatibility alias for projects that still expect to import the bundled
ldm package via ``import ldm``.
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

# Ensure the vendored taming-transformers package (providing `taming.*`) is importable.
_TAMING_TRANSFORMERS_ROOT = (Path(__file__).resolve().parent / "taming-transformers").resolve()
if _TAMING_TRANSFORMERS_ROOT.exists():
    _p = str(_TAMING_TRANSFORMERS_ROOT)
    if _p not in sys.path:
        # Do not prepend: this folder contains a top-level `main.py` which can
        # shadow this project's training entrypoint module name (`main`).
        sys.path.append(_p)

if "ldm" not in sys.modules:
    # Prefer the local bundled LDM implementation shipped with this repo.
    try:
        sys.modules["ldm"] = import_module(__name__ + ".ldm")  # davae.ldm
    except ModuleNotFoundError:
        # Legacy fallback for older codebases that used a different package name.
        try:
            sys.modules["ldm"] = import_module("LightningDiT.vae.ldm")
        except ModuleNotFoundError:
            # When the optional ldm tree is absent we simply skip the alias.
            pass

__all__ = []
