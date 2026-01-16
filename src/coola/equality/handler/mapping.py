r"""Implement handlers for mapping objects."""

from __future__ import annotations

__all__ = ["MappingSameKeysHandler", "MappingSameValuesHandler"]

import logging
from typing import TYPE_CHECKING, Any

from coola.equality.handler.base import BaseEqualityHandler
from coola.equality.handler.utils import check_recursion_depth

if TYPE_CHECKING:
    from collections.abc import Mapping

    from coola.equality.config import EqualityConfig

logger: logging.Logger = logging.getLogger(__name__)


class MappingSameKeysHandler(BaseEqualityHandler):
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
                missing_keys = keys1 - keys2
                additional_keys = keys2 - keys1
                logger.info(
                    f"mappings have different keys:\n"
                    f"missing keys: {sorted(missing_keys)}\n"
                    f"additional keys: {sorted(additional_keys)}"
                )
            return False
        return self._handle_next(actual, expected, config=config)


class MappingSameValuesHandler(BaseEqualityHandler):
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
            for key in actual:
                if not config.registry.objects_are_equal(actual[key], expected[key], config):
                    self._show_difference(actual, expected, config=config)
                    return False
            return self._handle_next(actual, expected, config=config)

    def _show_difference(
        self, actual: Mapping[Any, Any], expected: Mapping[Any, Any], config: EqualityConfig
    ) -> None:
        if config.show_difference:
            logger.info(
                f"mappings have at least one different value:\n"
                f"first mapping : {actual}\n"
                f"second mapping: {expected}"
            )
