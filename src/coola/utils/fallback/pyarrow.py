r"""Contain fallback implementations used when ``pyarrow`` dependency is
not available."""

from __future__ import annotations

__all__ = ["pyarrow"]

from types import ModuleType
from typing import Any

from coola.utils.imports import raise_error_pyarrow_missing


class Array:
    r"""Fake Array class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        raise_error_pyarrow_missing()


# Create a fake pyarrow package
pyarrow: ModuleType = ModuleType("pyarrow")
pyarrow.Array = Array
