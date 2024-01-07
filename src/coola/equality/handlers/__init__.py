r"""Contain the handlers to help to check if two objects are equal or
not.

The handlers are designed to work with Chain of Responsibility pattern.
"""

from __future__ import annotations

__all__ = [
    "AbstractEqualityHandler",
    "BaseEqualityHandler",
    "FalseHandler",
    "JaxArrayEqualHandler",
    "MappingSameKeysHandler",
    "MappingSameValuesHandler",
    "NumpyArrayEqualHandler",
    "ObjectEqualHandler",
    "SameAttributeHandler",
    "SameDTypeHandler",
    "SameLengthHandler",
    "SameObjectHandler",
    "SameShapeHandler",
    "SameTypeHandler",
    "SequenceSameValuesHandler",
    "TorchTensorEqualHandler",
    "TrueHandler",
]

from coola.equality.handlers.base import AbstractEqualityHandler, BaseEqualityHandler
from coola.equality.handlers.dtype import SameDTypeHandler
from coola.equality.handlers.jax_ import JaxArrayEqualHandler
from coola.equality.handlers.mapping import (
    MappingSameKeysHandler,
    MappingSameValuesHandler,
)
from coola.equality.handlers.native import (
    FalseHandler,
    ObjectEqualHandler,
    SameAttributeHandler,
    SameLengthHandler,
    SameObjectHandler,
    SameTypeHandler,
    TrueHandler,
)
from coola.equality.handlers.numpy_ import NumpyArrayEqualHandler
from coola.equality.handlers.sequence import SequenceSameValuesHandler
from coola.equality.handlers.shape import SameShapeHandler
from coola.equality.handlers.torch_ import TorchTensorEqualHandler
