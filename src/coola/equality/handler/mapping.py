r"""Implement handlers for mapping objects."""

from __future__ import annotations

__all__ = ["MappingSameKeyOrderHandler", "MappingSameKeysHandler", "MappingSameValuesHandler"]

from typing import TYPE_CHECKING, Any

from coola.equality.handler.base import BaseEqualityHandler
from coola.equality.handler.format import format_mapping_difference
from coola.equality.handler.mixin import HandlerEqualityMixin
from coola.equality.handler.utils import (
    check_recursion_depth,
    comparison_key,
    report_difference,
    track_comparison,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from coola.equality.config import EqualityConfig


class MappingSameKeysHandler(HandlerEqualityMixin, BaseEqualityHandler):
    r"""Check if the two objects have the same keys.

    This handler returns ``False`` if the two objects have different
    keys, otherwise it passes the inputs to the next handler. Keys are
    compared with their type, so ``1``, ``1.0`` and ``True`` are
    different keys.

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
        # Keys are compared with their type, like the elements of sets:
        # ``{1: "a"}`` and ``{True: "a"}`` are not the same mapping. With ``equal_nan=True``,
        # two NaN keys are the same key.
        keys1 = {comparison_key(key, config): key for key in actual}
        keys2 = {comparison_key(key, config): key for key in expected}
        if keys1.keys() != keys2.keys():
            if config.show_difference:
                report_difference(
                    config,
                    format_mapping_difference(
                        missing_keys={keys1[k] for k in keys1.keys() - keys2.keys()},
                        additional_keys={keys2[k] for k in keys2.keys() - keys1.keys()},
                    ),
                )
            return False
        return self._handle_next(actual, expected, config=config)


class MappingSameKeyOrderHandler(HandlerEqualityMixin, BaseEqualityHandler):
    r"""Check if the two objects have their keys in the same order.

    This handler returns ``False`` if the keys of the two objects are
    not iterated in the same order, otherwise it passes the inputs to
    the next handler. It is meant for mappings whose order is
    meaningful, like ``collections.OrderedDict``.

    Notes:
        This handler does not check that the two mappings have the
        same keys. Combine it with ``MappingSameKeysHandler``, which
        gives a more precise message for missing and additional keys.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import MappingSameKeyOrderHandler, TrueHandler
        >>> config = EqualityConfig()
        >>> handler = MappingSameKeyOrderHandler(next_handler=TrueHandler())
        >>> handler.handle({"a": 1, "b": 2}, {"a": 1, "b": 2}, config)
        True
        >>> handler.handle({"a": 1, "b": 2}, {"b": 2, "a": 1}, config)
        False

        ```
    """

    def handle(
        self,
        actual: Mapping[Any, Any],
        expected: Mapping[Any, Any],
        config: EqualityConfig,
    ) -> bool:
        actual_keys = list(actual)
        expected_keys = list(expected)
        if [comparison_key(key, config) for key in actual_keys] != [
            comparison_key(key, config) for key in expected_keys
        ]:
            if config.show_difference:
                report_difference(
                    config,
                    format_mapping_difference(different_key_order=(actual_keys, expected_keys)),
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
        with (
            check_recursion_depth(config),
            track_comparison(config, actual, expected) as in_progress,
        ):
            if in_progress:
                # Cycle: the comparison of this pair is already in progress.
                return True
            for key in actual:
                value1, value2 = actual[key], _get_value(expected, key, config)
                if not config.registry.objects_are_equal(value1, value2, config):
                    if config.show_difference:
                        report_difference(
                            config,
                            format_mapping_difference(different_value_key=key),
                            path=(key,),
                        )
                    return False
            return self._handle_next(actual, expected, config=config)


def _get_value(mapping: Mapping[Any, Any], key: Any, config: EqualityConfig) -> Any:
    r"""Return ``mapping[key]``, a NaN key matching another NaN key if
    ``config.equal_nan`` is ``True``."""
    try:
        return mapping[key]
    except KeyError:
        if not config.equal_nan:
            raise
        # Two NaNs are different objects with different hashes: find the key by its comparison key.
        target = comparison_key(key, config)
        for other in mapping:
            if comparison_key(other, config) == target:
                return mapping[other]
        raise
