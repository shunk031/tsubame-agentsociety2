"""Make the scripts in ``scripts/`` importable by name.

They are standalone entry points rather than a package, so there is nothing to
install. Both modules under test import only the standard library, which is why
the suite runs without the TSUBAME environment:

    uv run --with pytest --no-project --python 3.12 pytest
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
