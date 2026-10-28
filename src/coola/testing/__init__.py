r"""Provide ``pytest`` marks and fixtures to test code that has optional
dependencies (e.g. NumPy, pandas, PyTorch).

The marks are defined in ``coola.testing.marks``. Import one from there
and use it as a test decorator, e.g. ``from coola.testing.marks import
numpy_available`` then ``@numpy_available`` skips the test unless NumPy
is installed. The fixtures are defined in ``coola.testing.fixtures``:
load them with ``pytest_plugins = ["coola.testing.fixtures"]`` in the
top-level ``conftest.py``.

``pytest`` is required: install ``coola`` with the ``testing`` extra
(``pip install coola[testing]``).
"""

from __future__ import annotations

__all__: list[str] = []
