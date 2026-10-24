r"""Composable equality handlers used by tester implementations.

Handlers are combined in chains so each check can either short-circuit
or delegate to the next check in the sequence.
"""

from __future__ import annotations

__all__ = [
    "AllCloseNanHandler",
    "AttributeName",
    "BaseEqualityHandler",
    "DataclassSameFieldsHandler",
    "DifferenceCloseHandler",
    "EqualHandler",
    "EqualNanHandler",
    "FalseHandler",
    "HandlerCycleError",
    "HandlerEqualityMixin",
    "JaxArrayEqualHandler",
    "MappingSameKeyOrderHandler",
    "MappingSameKeysHandler",
    "MappingSameValuesHandler",
    "MaxDepthExceededError",
    "NanEqualHandler",
    "NumpyArrayEqualHandler",
    "ObjectEqualHandler",
    "ObjectSameAttributesHandler",
    "PandasDataFrameEqualHandler",
    "PandasExtensionArrayEqualHandler",
    "PandasIndexEqualHandler",
    "PandasSeriesEqualHandler",
    "PolarsDataFrameEqualHandler",
    "PolarsLazyFrameEqualHandler",
    "PolarsSeriesEqualHandler",
    "PyarrowEqualHandler",
    "PydanticModelSameFieldsHandler",
    "SameAttributeHandler",
    "SameDTypeHandler",
    "SameDataHandler",
    "SameLengthHandler",
    "SameObjectHandler",
    "SameShapeHandler",
    "SameTypeHandler",
    "ScalarEqualHandler",
    "SequenceSameValuesHandler",
    "SetSameValuesHandler",
    "TolerantEqualHandler",
    "TorchTensorEqualHandler",
    "TorchTensorSameDeviceHandler",
    "TrueHandler",
    "check_recursion_depth",
    "create_chain",
    "get_object_attributes",
    "handlers_are_equal",
    "report_difference",
    "track_comparison",
]

from typing import TYPE_CHECKING

from coola.equality.handler.allclose import AllCloseNanHandler
from coola.equality.handler.base import BaseEqualityHandler, HandlerCycleError
from coola.equality.handler.data import SameDataHandler
from coola.equality.handler.dataclass import DataclassSameFieldsHandler
from coola.equality.handler.difference import DifferenceCloseHandler
from coola.equality.handler.dtype import SameDTypeHandler
from coola.equality.handler.equal import EqualHandler, EqualNanHandler
from coola.equality.handler.mapping import (
    MappingSameKeyOrderHandler,
    MappingSameKeysHandler,
    MappingSameValuesHandler,
)
from coola.equality.handler.mixin import HandlerEqualityMixin
from coola.equality.handler.native import (
    FalseHandler,
    ObjectEqualHandler,
    SameAttributeHandler,
    SameLengthHandler,
    SameObjectHandler,
    SameTypeHandler,
    TrueHandler,
)
from coola.equality.handler.object import ObjectSameAttributesHandler, get_object_attributes
from coola.equality.handler.pydantic import PydanticModelSameFieldsHandler
from coola.equality.handler.scalar import NanEqualHandler, ScalarEqualHandler
from coola.equality.handler.sequence import SequenceSameValuesHandler
from coola.equality.handler.set import SetSameValuesHandler
from coola.equality.handler.shape import SameShapeHandler
from coola.equality.handler.tolerant import TolerantEqualHandler
from coola.equality.handler.utils import (
    AttributeName,
    MaxDepthExceededError,
    check_recursion_depth,
    create_chain,
    handlers_are_equal,
    report_difference,
    track_comparison,
)
from coola.utils.imports import lazy_exports

if TYPE_CHECKING:
    from coola.equality.handler.jax import JaxArrayEqualHandler
    from coola.equality.handler.numpy import NumpyArrayEqualHandler
    from coola.equality.handler.pandas import (
        PandasDataFrameEqualHandler,
        PandasExtensionArrayEqualHandler,
        PandasIndexEqualHandler,
        PandasSeriesEqualHandler,
    )
    from coola.equality.handler.polars import (
        PolarsDataFrameEqualHandler,
        PolarsLazyFrameEqualHandler,
        PolarsSeriesEqualHandler,
    )
    from coola.equality.handler.pyarrow import PyarrowEqualHandler
    from coola.equality.handler.torch import (
        TorchTensorEqualHandler,
        TorchTensorSameDeviceHandler,
    )


# The names defined in the modules that import an optional library are imported the first time they
# are used, so the library is not imported with the package.
_LAZY_EXPORTS = {
    "JaxArrayEqualHandler": "coola.equality.handler.jax",
    "NumpyArrayEqualHandler": "coola.equality.handler.numpy",
    "PandasDataFrameEqualHandler": "coola.equality.handler.pandas",
    "PandasExtensionArrayEqualHandler": "coola.equality.handler.pandas",
    "PandasIndexEqualHandler": "coola.equality.handler.pandas",
    "PandasSeriesEqualHandler": "coola.equality.handler.pandas",
    "PolarsDataFrameEqualHandler": "coola.equality.handler.polars",
    "PolarsLazyFrameEqualHandler": "coola.equality.handler.polars",
    "PolarsSeriesEqualHandler": "coola.equality.handler.polars",
    "PyarrowEqualHandler": "coola.equality.handler.pyarrow",
    "TorchTensorEqualHandler": "coola.equality.handler.torch",
    "TorchTensorSameDeviceHandler": "coola.equality.handler.torch",
}
__getattr__, __dir__ = lazy_exports(__name__, _LAZY_EXPORTS, globals())
