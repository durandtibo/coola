r"""Contain fallback implementations used when ``urllib3`` dependency is
not available."""

from __future__ import annotations

__all__ = ["Retry"]

from typing import Any

from feu.imports import raise_error_urllib3_missing


class Retry:
    r"""Create a fake Retry class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        raise_error_urllib3_missing()
