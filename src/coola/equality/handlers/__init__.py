r"""Contain the handlers to help to check if two objects are equal or
not.

The handlers are designed to work with Chain of Responsibility pattern.
"""

from __future__ import annotations

__all__ = [
    "AbstractEqualityHandler",
    "ArraySameDTypeHandler",
    "BaseEqualityHandler",
    "FalseHandler",
    "ObjectEqualHandler",
    "SameKeysHandler",
    "SameLengthHandler",
    "SameObjectHandler",
    "SameTypeHandler",
    "TrueHandler",
]

from coola.equality.handlers.array import ArraySameDTypeHandler
from coola.equality.handlers.base import AbstractEqualityHandler, BaseEqualityHandler
from coola.equality.handlers.native import (
    FalseHandler,
    ObjectEqualHandler,
    SameKeysHandler,
    SameLengthHandler,
    SameObjectHandler,
    SameTypeHandler,
    TrueHandler,
)
