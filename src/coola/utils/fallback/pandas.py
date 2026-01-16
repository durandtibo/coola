r"""Contain fallback implementations used when ``pandas`` dependency is
not available."""

from __future__ import annotations

__all__ = ["pandas"]

from types import ModuleType
from typing import Any

from coola.utils.imports import raise_error_pandas_missing


class FakeClass:
    r"""Fake class that raises an error because pandas is not installed.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Raises:
        RuntimeError: pandas is required for this functionality.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        raise_error_pandas_missing()


# Create a fake pandas package
pandas: ModuleType = ModuleType("pandas")
pandas.DataFrame = FakeClass
pandas.Series = FakeClass
