r"""Implement a handler to check if two pydantic models are equal."""

from __future__ import annotations

__all__ = ["PydanticModelSameFieldsHandler"]

import logging
from typing import TYPE_CHECKING, Any

from coola.equality.handler.base import BaseEqualityHandler
from coola.equality.handler.mixin import HandlerEqualityMixin
from coola.equality.handler.utils import (
    AttributeName,
    check_recursion_depth,
    report_difference,
    track_comparison,
)
from coola.utils.safe import safe_str

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig

logger: logging.Logger = logging.getLogger(__name__)


class PydanticModelSameFieldsHandler(HandlerEqualityMixin, BaseEqualityHandler):
    r"""Check if two pydantic models have the same field values.

    Only the declared fields (``model_fields``) are compared, through
    the registry, so the tolerance, ``equal_nan`` and the testers of
    the field types are used. The other entries of the instance
    ``__dict__`` (for example the value of a
    ``functools.cached_property`` that was read on one model only) and
    the private attributes are ignored, as they are not part of the
    state that the model validates and serializes. This handler expects
    two instances of the same model class.

    Example:
        ```pycon
        >>> from pydantic import BaseModel
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import PydanticModelSameFieldsHandler, TrueHandler
        >>> class Point(BaseModel):
        ...     x: float
        ...     y: float
        ...
        >>> config = EqualityConfig()
        >>> handler = PydanticModelSameFieldsHandler(next_handler=TrueHandler())
        >>> handler.handle(Point(x=1.0, y=2.0), Point(x=1.0, y=2.0), config)
        True
        >>> handler.handle(Point(x=1.0, y=2.0), Point(x=1.0, y=3.0), config)
        False

        ```
    """

    def handle(self, actual: Any, expected: Any, config: EqualityConfig) -> bool:
        with (
            check_recursion_depth(config),
            track_comparison(config, actual, expected) as in_progress,
        ):
            if in_progress:
                # Cycle: the comparison of this pair is already in progress.
                return True
            # ``type(actual).model_fields``: the access from an instance is deprecated.
            for name in type(actual).model_fields:
                value1 = getattr(actual, name)
                value2 = getattr(expected, name)
                if not config.registry.objects_are_equal(value1, value2, config):
                    if config.show_difference:
                        report_difference(
                            logger,
                            config,
                            "objects have different %s: %s vs %s",
                            name,
                            safe_str(value1),
                            safe_str(value2),
                            extra={"coola_path": (AttributeName(name),)},
                        )
                    return False
            return self._handle_next(actual, expected, config=config)
