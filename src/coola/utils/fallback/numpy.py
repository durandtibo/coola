r"""Contain fallback implementations used when ``numpy`` dependency is
not available."""

from __future__ import annotations

__all__ = ["numpy"]

from types import ModuleType
from typing import Any

from coola.utils.imports import raise_error_numpy_missing


class FakeClass:
    r"""Fake class that raises an error because numpy is not installed.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Raises:
        RuntimeError: numpy is required for this functionality.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        raise_error_numpy_missing()


# Create a fake numpy package
numpy: ModuleType = ModuleType("numpy")
numpy.ma = ModuleType("numpy.ma")
numpy.ma.MaskedArray = FakeClass
numpy.ndarray = FakeClass
