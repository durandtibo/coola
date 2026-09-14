r"""Implement handlers for mapping objects."""

from __future__ import annotations

__all__ = ["MappingSameKeysHandler", "MappingSameValuesHandler"]

import logging
from typing import TYPE_CHECKING, Any

from coola.equality.handler.base import BaseEqualityHandler
from coola.equality.handler.format import format_mapping_difference
from coola.equality.handler.mixin import HandlerEqualityMixin
from coola.equality.handler.utils import check_recursion_depth

if TYPE_CHECKING:
    from collections.abc import Mapping

    from coola.equality.config import EqualityConfig

logger: logging.Logger = logging.getLogger(__name__)


class MappingSameKeysHandler(HandlerEqualityMixin, BaseEqualityHandler):
    r"""Check if the two objects have the same keys.

    This handler returns ``False`` if the two objects have different
    keys, otherwise it passes the inputs to the next handler.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import MappingSameKeysHandler
        >>> config = EqualityConfig()
        >>> handler = MappingSameKeysHandler()
        >>> handler.handle({"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 1}, config)
        False

        ```
    """

    def handle(
        self,
        actual: Mapping[Any, Any],
        expected: Mapping[Any, Any],
        config: EqualityConfig,
    ) -> bool:
        keys1 = set(actual.keys())
        keys2 = set(expected.keys())
        if keys1 != keys2:
            if config.show_difference:
                logger.info(
                    format_mapping_difference(
                        missing_keys=keys1 - keys2, additional_keys=keys2 - keys1
                    )
                )
            return False
        return self._handle_next(actual, expected, config=config)


class MappingSameValuesHandler(HandlerEqualityMixin, BaseEqualityHandler):
    r"""Check if the key-value pairs in the first mapping are in the
    second mapping.

    This handler returns ``False`` if the one of the key-value pair in
    the first mapping is not in the second mapping, otherwise it
    passes the inputs to the next handler.

    Notes:
        This handler assumes that all the keys in the first mapping are
        also in the second mapping. The second mapping can have more
        keys. To check if two mappings are equal, you can combine this
        handler with ``MappingSameKeysHandler``.

        Within a single ``handle`` call, the result of comparing a
        given pair of values is cached by object identity (``id()``),
        so repeated or interned values that appear under several keys
        are compared only once, avoiding redundant recursive
        ``objects_are_equal`` calls for highly repetitive mappings.

    Warning:
        This handler does not check the assumption above itself: it
        accesses ``expected[key]`` directly for every key in ``actual``
        without a ``.get``/``try``-``except``. If this handler is used
        standalone (or as the first handler in a chain, without
        ``MappingSameKeysHandler`` preceding it) and ``actual`` has a
        key that is missing from ``expected``, ``handle`` raises
        ``KeyError`` instead of returning ``False``. Always chain this
        handler after ``MappingSameKeysHandler`` unless the missing-key
        case is otherwise guaranteed not to occur.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import MappingSameValuesHandler, TrueHandler
        >>> config = EqualityConfig()
        >>> handler = MappingSameValuesHandler(next_handler=TrueHandler())
        >>> handler.handle({"a": 1, "b": 2}, {"a": 1, "b": 2}, config)
        True
        >>> handler.handle({"a": 1, "b": 2}, {"a": 1, "b": 3}, config)
        False

        ```
    """

    def handle(
        self,
        actual: Mapping[Any, Any],
        expected: Mapping[Any, Any],
        config: EqualityConfig,
    ) -> bool:
        with check_recursion_depth(config):
            cache: dict[tuple[int, int], bool] = {}
            for key in actual:
                value1, value2 = actual[key], expected[key]
                cache_key = (id(value1), id(value2))
                are_equal = cache.get(cache_key)
                if are_equal is None:
                    are_equal = config.registry.objects_are_equal(value1, value2, config)
                    cache[cache_key] = are_equal
                if not are_equal:
                    if config.show_difference:
                        logger.info(format_mapping_difference(different_value_key=key))
                    return False
            return self._handle_next(actual, expected, config=config)
