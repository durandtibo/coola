r"""Composable equality handlers used by tester implementations.

Handlers are combined in chains so each check can either short-circuit
or delegate to the next check in the sequence.
"""

from __future__ import annotations

__all__ = [
    "AllCloseNanHandler",
    "AttributeName",
    "BaseEqualityHandler",
    "FalseHandler",
    "HandlerCycleError",
    "HandlerEqualityMixin",
    "MaxDepthExceededError",
    "ObjectEqualHandler",
    "PandasDataFrameEqualHandler",
    "PandasExtensionArrayEqualHandler",
    "PandasIndexEqualHandler",
    "PandasSeriesEqualHandler",
    "PolarsDataFrameEqualHandler",
    "PolarsLazyFrameEqualHandler",
    "PolarsSeriesEqualHandler",
    "PyarrowEqualHandler",
    "SameAttributeHandler",
    "SameLengthHandler",
    "SameObjectHandler",
    "SameTypeHandler",
    "TrueHandler",
    "check_recursion_depth",
    "comparison_key",
    "create_chain",
    "get_object_attributes",
    "handlers_are_equal",
    "is_exact_scalar_sequence",
    "report_difference",
    "track_comparison",
]

from typing import TYPE_CHECKING

from coola.equality.handler.allclose import AllCloseNanHandler
from coola.equality.handler.base import BaseEqualityHandler, HandlerCycleError
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
from coola.equality.handler.object import get_object_attributes
from coola.equality.handler.utils import (
    AttributeName,
    MaxDepthExceededError,
    check_recursion_depth,
    comparison_key,
    create_chain,
    handlers_are_equal,
    is_exact_scalar_sequence,
    report_difference,
    track_comparison,
)
from coola.utils.imports import lazy_exports

if TYPE_CHECKING:
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


# The names defined in the modules that import an optional library are imported the first time they
# are used, so the library is not imported with the package.
_LAZY_EXPORTS = {
    "PandasDataFrameEqualHandler": "coola.equality.handler.pandas",
    "PandasExtensionArrayEqualHandler": "coola.equality.handler.pandas",
    "PandasIndexEqualHandler": "coola.equality.handler.pandas",
    "PandasSeriesEqualHandler": "coola.equality.handler.pandas",
    "PolarsDataFrameEqualHandler": "coola.equality.handler.polars",
    "PolarsLazyFrameEqualHandler": "coola.equality.handler.polars",
    "PolarsSeriesEqualHandler": "coola.equality.handler.polars",
    "PyarrowEqualHandler": "coola.equality.handler.pyarrow",
}
__getattr__, __dir__ = lazy_exports(__name__, _LAZY_EXPORTS, globals())
