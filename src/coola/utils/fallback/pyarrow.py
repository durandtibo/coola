r"""Contain fallback implementations used when ``pyarrow`` dependency is
not available."""

from __future__ import annotations

__all__ = ["pyarrow"]

from types import ModuleType
from typing import Any

from coola.utils.imports import raise_error_pyarrow_missing


class FakeClass:
    r"""Fake class that raises an error because pyarrow is not installed.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Raises:
        RuntimeError: pyarrow is required for this functionality.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        raise_error_pyarrow_missing()


# Create a fake pyarrow package
pyarrow: ModuleType = ModuleType("pyarrow")
pyarrow.Array = FakeClass
