r"""Implement a handler to check if two objects have the same
attributes."""

from __future__ import annotations

__all__ = ["ObjectSameAttributesHandler", "get_object_attributes"]

import functools
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


def get_object_attributes(obj: object) -> dict[str, Any]:
    r"""Get the attributes of an object that define its state.

    The attributes are the entries of ``vars(obj)``, the values of the
    slots (``__slots__``) of its class and of its parent classes (a
    slot that is not set is ignored), the ``args`` of an exception and
    the ``func``, ``args`` and ``keywords`` of a
    ``functools.partial``. A slot or an entry of ``vars`` takes
    precedence over the ``args`` of an exception.

    Args:
        obj: The object.

    Returns:
        The attributes, by name.

    Example:
        ```pycon
        >>> from coola.equality.handler import get_object_attributes
        >>> class Point:
        ...     def __init__(self, x, y):
        ...         self.x = x
        ...         self.y = y
        ...
        >>> get_object_attributes(Point(1, 2))
        {'x': 1, 'y': 2}
        >>> get_object_attributes(ValueError("a"))
        {'args': ('a',)}

        ```
    """
    attributes: dict[str, Any] = {}
    if isinstance(obj, BaseException):
        attributes["args"] = obj.args
    if isinstance(obj, functools.partial):
        attributes.update(func=obj.func, args=obj.args, keywords=obj.keywords)
    for cls in reversed(type(obj).__mro__):
        slots = cls.__dict__.get("__slots__", ())
        for name in (slots,) if isinstance(slots, str) else slots:
            if name in {"__dict__", "__weakref__"} or not hasattr(obj, name):
                continue
            attributes[name] = getattr(obj, name)
    attributes.update(getattr(obj, "__dict__", {}))
    return attributes


class ObjectSameAttributesHandler(HandlerEqualityMixin, BaseEqualityHandler):
    r"""Check if two objects have the same attributes.

    The attributes are the ones returned by ``get_object_attributes``
    (``vars``, slots, the ``args`` of an exception, the content of a
    ``functools.partial``). The two objects must have the same
    attribute names and equal values, which are compared through the
    registry, so the tolerance and ``equal_nan`` apply to the values.
    This handler expects two instances of the same class.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import ObjectSameAttributesHandler, TrueHandler
        >>> class Point:
        ...     def __init__(self, x, y):
        ...         self.x = x
        ...         self.y = y
        ...
        >>> config = EqualityConfig()
        >>> handler = ObjectSameAttributesHandler(next_handler=TrueHandler())
        >>> handler.handle(Point(1.0, 2.0), Point(1.0, 2.0), config)
        True
        >>> handler.handle(Point(1.0, 2.0), Point(1.0, 3.0), config)
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
            attributes1 = get_object_attributes(actual)
            attributes2 = get_object_attributes(expected)
            if attributes1.keys() != attributes2.keys():
                if config.show_difference:
                    report_difference(
                        logger,
                        config,
                        "objects have different attributes: %s vs %s",
                        sorted(attributes1),
                        sorted(attributes2),
                    )
                return False
            for name, value1 in attributes1.items():
                value2 = attributes2[name]
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
