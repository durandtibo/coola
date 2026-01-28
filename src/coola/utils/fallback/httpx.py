r"""Contain fallback implementations used when ``httpx`` dependency is
not available."""

from __future__ import annotations

__all__ = ["httpx"]

from types import ModuleType
from typing import Any

from coola.utils.imports import raise_httpx_missing_error


class FakeClass:
    r"""Fake class that raises an error because httpx is not installed.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Raises:
        RuntimeError: httpx is required for this functionality.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        raise_httpx_missing_error()


# Create a fake httpx package
httpx: ModuleType = ModuleType("httpx")
httpx.Response = FakeClass
