r"""Implement a handler to check if two dataclass instances are
equal."""

from __future__ import annotations

__all__ = ["DataclassSameFieldsHandler"]

import dataclasses
import logging
from typing import TYPE_CHECKING, Any

from coola.equality.handler.base import BaseEqualityHandler
from coola.equality.handler.mixin import HandlerEqualityMixin
from coola.equality.handler.utils import check_recursion_depth, report_difference

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig

logger: logging.Logger = logging.getLogger(__name__)


class DataclassSameFieldsHandler(HandlerEqualityMixin, BaseEqualityHandler):
    r"""Check if two dataclass instances have the same field values.

    The fields are compared through the registry, so the tolerance,
    ``equal_nan`` and the testers of the field types are used. The
    fields with ``compare=False`` are ignored. This handler expects
    two instances of the same dataclass.

    Example:
        ```pycon
        >>> from dataclasses import dataclass
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import DataclassSameFieldsHandler, TrueHandler
        >>> @dataclass
        ... class Point:
        ...     x: float
        ...     y: float
        ...
        >>> config = EqualityConfig()
        >>> handler = DataclassSameFieldsHandler(next_handler=TrueHandler())
        >>> handler.handle(Point(1.0, 2.0), Point(1.0, 2.0), config)
        True
        >>> handler.handle(Point(1.0, 2.0), Point(1.0, 3.0), config)
        False

        ```
    """

    def handle(self, actual: Any, expected: Any, config: EqualityConfig) -> bool:
        with check_recursion_depth(config):
            for field in dataclasses.fields(actual):
                if not field.compare:
                    continue
                value1 = getattr(actual, field.name)
                value2 = getattr(expected, field.name)
                if not config.registry.objects_are_equal(value1, value2, config):
                    if config.show_difference:
                        report_difference(
                            logger,
                            config,
                            "objects have different %s: %s vs %s",
                            field.name,
                            value1,
                            value2,
                        )
                    return False
            return self._handle_next(actual, expected, config=config)
