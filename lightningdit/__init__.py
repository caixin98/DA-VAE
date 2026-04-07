# Ensure LightningDiT package absolute imports (e.g., models.*, vae.*) work when running via -m or scripts
# without installing the package.
import sys as _sys
from pathlib import Path as _Path

_pkg_root = _Path(__file__).resolve().parent
if str(_pkg_root) not in _sys.path:
    _sys.path.insert(0, str(_pkg_root))
