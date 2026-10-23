r"""Contain display shared helpers."""

from __future__ import annotations

__all__ = [
    "BaseDisplayMixin",
    "InlineDisplayMixin",
    "MultilineDisplayMixin",
    "repr_pydantic_model",
    "str_pydantic_model",
]

from typing import TYPE_CHECKING

from coola.display.mixin import (
    BaseDisplayMixin,
    InlineDisplayMixin,
    MultilineDisplayMixin,
)
from coola.utils.imports import lazy_exports

if TYPE_CHECKING:
    from coola.display.pydantic import repr_pydantic_model, str_pydantic_model


# The names defined in the modules that import an optional library are imported the first time they
# are used, so the library is not imported with the package.
_LAZY_EXPORTS = {
    "repr_pydantic_model": "coola.display.pydantic",
    "str_pydantic_model": "coola.display.pydantic",
}
__getattr__, __dir__ = lazy_exports(__name__, _LAZY_EXPORTS, globals())
