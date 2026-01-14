r"""Contain the handler to help to check if two objects are equal or not.

The handler are designed to work with the Chain of Responsibility
pattern.
"""

from __future__ import annotations

__all__ = [
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
    "PolarsLazyFrameEqualHandler",
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
    "TorchTensorSameDeviceHandler",
    "TrueHandler",
    "handlers_are_equal",
]

from coola.equality.handler.base import BaseEqualityHandler
from coola.equality.handler.data import SameDataHandler
from coola.equality.handler.dtype import SameDTypeHandler
from coola.equality.handler.equal import EqualHandler, EqualNanHandler
from coola.equality.handler.jax import JaxArrayEqualHandler
from coola.equality.handler.mapping import (
    MappingSameKeysHandler,
    MappingSameValuesHandler,
)
from coola.equality.handler.native import (
    FalseHandler,
    ObjectEqualHandler,
    SameAttributeHandler,
    SameLengthHandler,
    SameObjectHandler,
    SameTypeHandler,
    TrueHandler,
)
from coola.equality.handler.numpy import NumpyArrayEqualHandler
from coola.equality.handler.pandas import (
    PandasDataFrameEqualHandler,
    PandasSeriesEqualHandler,
)
from coola.equality.handler.polars import (
    PolarsDataFrameEqualHandler,
    PolarsLazyFrameEqualHandler,
    PolarsSeriesEqualHandler,
)
from coola.equality.handler.pyarrow import PyarrowEqualHandler
from coola.equality.handler.scalar import NanEqualHandler, ScalarEqualHandler
from coola.equality.handler.sequence import SequenceSameValuesHandler
from coola.equality.handler.shape import SameShapeHandler
from coola.equality.handler.torch import (
    TorchTensorEqualHandler,
    TorchTensorSameDeviceHandler,
)
from coola.equality.handler.utils import handlers_are_equal
