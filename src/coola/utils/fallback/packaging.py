r"""Contain fallback implementations used when ``packaging`` dependency
is not available."""

from __future__ import annotations

__all__ = ["Version"]

from typing import Any

from coola.utils.imports import raise_error_packaging_missing


class Version:
    r"""Create a fake Version class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        raise_error_packaging_missing()
