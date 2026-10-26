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
