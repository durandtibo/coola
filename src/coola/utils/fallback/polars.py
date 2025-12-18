r"""Contain fallback implementations used when ``polars`` dependency is
not available."""

from __future__ import annotations

__all__ = ["polars"]

from types import ModuleType
from typing import Any

from coola.utils.imports import raise_error_polars_missing


class FakeClass:
    r"""Fake class that raises an error because polars is not installed.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Raises:
        RuntimeError: polars is required for this functionality.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        raise_error_polars_missing()


# Create a fake polars package
polars: ModuleType = ModuleType("polars")
polars.DataFrame = FakeClass
polars.LazyFrame = FakeClass
polars.Series = FakeClass
