r"""Implement some handlers for native python objects."""

from __future__ import annotations

__all__ = ["MappingSameKeysHandler", "MappingSameValuesHandler"]

import logging
from typing import TYPE_CHECKING

from coola.equality.handlers.base import AbstractEqualityHandler

if TYPE_CHECKING:
    from collections.abc import Mapping

    from coola.equality.config import EqualityConfig

logger = logging.getLogger(__name__)


class MappingSameKeysHandler(AbstractEqualityHandler):
    r"""Check if the two objects have the same keys.

    This handler returns ``False`` if the two objects have different
    keys, otherwise it passes the inputs to the next handler.

    Example usage:

    ```pycon
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.handlers import MappingSameKeysHandler
    >>> from coola.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> handler = MappingSameKeysHandler()
    >>> handler.handle({"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 1}, config)
    False

    ```
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def handle(
        self,
        object1: Mapping,
        object2: Mapping,
        config: EqualityConfig,
    ) -> bool:
        if set(object1.keys()) != set(object2.keys()):
            if config.show_difference:
                logger.info(
                    f"mappings have different keys:\n"
                    f"object1 keys: {sorted(set(object1.keys()))}\n"
                    f"object2 keys: {sorted(set(object2.keys()))}"
                )
            return False
        return self._handle_next(object1=object1, object2=object2, config=config)


class MappingSameValuesHandler(AbstractEqualityHandler):
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

    Example usage:

    ```pycon
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.handlers import MappingSameValuesHandler, TrueHandler
    >>> from coola.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> handler = MappingSameValuesHandler(next_handler=TrueHandler())
    >>> handler.handle({"a": 1, "b": 2}, {"a": 1, "b": 2}, config)
    True
    >>> handler.handle({"a": 1, "b": 2}, {"a": 1, "b": 3}, config)
    False

    ```
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def handle(
        self,
        object1: Mapping,
        object2: Mapping,
        config: EqualityConfig,
    ) -> bool:
        for key in object1:
            if not config.tester.equal(object1[key], object2[key], config.show_difference):
                self._show_difference(object1=object1, object2=object2, config=config)
                return False
        return self._handle_next(object1=object1, object2=object2, config=config)

    def _show_difference(self, object1: Mapping, object2: Mapping, config: EqualityConfig) -> None:
        if config.show_difference:
            logger.info(
                f"mappings have at least one different value:\n"
                f"first mapping : {object1}\n"
                f"second mapping: {object2}"
            )
