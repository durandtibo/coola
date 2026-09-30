r"""Provide ``pytest`` fixtures to skip or require tests based on the
availability of optional dependencies (e.g. NumPy, pandas, PyTorch).

The fixtures are defined in ``coola.testing.fixtures``. Import one from
there and use it as a test decorator, e.g. ``from coola.testing.fixtures
import numpy_available`` then ``@numpy_available`` skips the test unless
NumPy is installed.
"""

from __future__ import annotations

__all__: list[str] = []
