r"""Contain fallback implementations used when ``pydantic`` dependency
is not available."""

from __future__ import annotations

__all__ = ["pydantic"]

from types import ModuleType
from typing import Any

from coola.utils.imports import raise_pydantic_missing_error


class FakeClass:
    r"""Fake class that raises an error because pydantic is not
    installed.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Raises:
        RuntimeError: pydantic is required for this functionality.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        raise_pydantic_missing_error()


# Create a fake pydantic package
pydantic: ModuleType = ModuleType("pydantic")
pydantic.BaseModel = FakeClass
pydantic.SecretBytes = FakeClass
pydantic.SecretStr = FakeClass
