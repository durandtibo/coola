r"""Provide helpers to display objects and to write to the terminal.

The package has three kinds of content:

- Mixins: ``InlineDisplayMixin``, ``MultilineDisplayMixin`` and
  ``BaseDisplayMixin`` give a class a consistent ``__repr__`` and
  ``__str__``. A class only overrides ``_display_fields``.
- Pydantic formatting: ``str_pydantic_model`` and ``repr_pydantic_model``
  format a pydantic model without exposing its secrets. They need
  ``pydantic``, which is imported the first time they are used.
- Terminal output: ``coola.display.rich`` and ``coola.display.colorlog``
  configure logging and print with ``rich`` and ``colorlog``. A function
  raises ``MissingPackageError`` when its library is not installed.
"""

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
