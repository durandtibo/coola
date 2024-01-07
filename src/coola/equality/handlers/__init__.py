r"""Contain the handlers to help to check if two objects are equal or
not.

The handlers are designed to work with Chain of Responsibility pattern.
"""

from __future__ import annotations

__all__ = [
    "AbstractEqualityHandler",
    "ArraySameDTypeHandler",
    "ArraySameShapeHandler",
    "BaseEqualityHandler",
    "FalseHandler",
    "MappingSameKeysHandler",
    "MappingSameValuesHandler",
    "ObjectEqualHandler",
    "SameLengthHandler",
    "SameObjectHandler",
    "SameTypeHandler",
    "SequenceSameValuesHandler",
    "TensorEqualHandler",
    "TrueHandler",
]

from coola.equality.handlers.array import ArraySameDTypeHandler, ArraySameShapeHandler
from coola.equality.handlers.base import AbstractEqualityHandler, BaseEqualityHandler
from coola.equality.handlers.mapping import (
    MappingSameKeysHandler,
    MappingSameValuesHandler,
)
from coola.equality.handlers.native import (
    FalseHandler,
    ObjectEqualHandler,
    SameLengthHandler,
    SameObjectHandler,
    SameTypeHandler,
    TrueHandler,
)
from coola.equality.handlers.sequence import SequenceSameValuesHandler
from coola.equality.handlers.torch_ import TensorEqualHandler
