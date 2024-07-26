r"""Contain the handlers to help to check if two objects are equal or
not.

The handlers are designed to work with Chain of Responsibility pattern.
"""

from __future__ import annotations

__all__ = [
    "AbstractEqualityHandler",
    "BaseEqualityHandler",
    "EqualHandler",
    "EqualNanHandler",
    "FalseHandler",
    "JaxArrayEqualHandler",
    "MappingSameKeysHandler",
    "MappingSameValuesHandler",
    "NanEqualHandler",
    "NumpyArrayEqualHandler",
    "ObjectEqualHandler",
    "PandasDataFrameEqualHandler",
    "PandasSeriesEqualHandler",
    "PolarsDataFrameEqualHandler",
    "PolarsSeriesEqualHandler",
    "PyarrowEqualHandler",
    "SameAttributeHandler",
    "SameDTypeHandler",
    "SameDataHandler",
    "SameLengthHandler",
    "SameObjectHandler",
    "SameShapeHandler",
    "SameTypeHandler",
    "ScalarEqualHandler",
    "SequenceSameValuesHandler",
    "TorchTensorEqualHandler",
    "TrueHandler",
]

from coola.equality.handlers.base import AbstractEqualityHandler, BaseEqualityHandler
from coola.equality.handlers.data import SameDataHandler
from coola.equality.handlers.dtype import SameDTypeHandler
from coola.equality.handlers.equal import EqualHandler, EqualNanHandler
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
from coola.equality.handlers.pandas_ import (
    PandasDataFrameEqualHandler,
    PandasSeriesEqualHandler,
)
from coola.equality.handlers.polars_ import (
    PolarsDataFrameEqualHandler,
    PolarsSeriesEqualHandler,
)
from coola.equality.handlers.pyarrow_ import PyarrowEqualHandler
from coola.equality.handlers.scalar import NanEqualHandler, ScalarEqualHandler
from coola.equality.handlers.sequence import SequenceSameValuesHandler
from coola.equality.handlers.shape import SameShapeHandler
from coola.equality.handlers.torch_ import TorchTensorEqualHandler
