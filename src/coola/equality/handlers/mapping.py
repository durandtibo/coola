r"""Implement some handlers for native python objects."""

from __future__ import annotations

__all__ = ["MappingSameKeysHandler"]

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

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def handle(
        self,
        object1: Mapping,
        object2: Mapping,
        config: EqualityConfig,
    ) -> bool | None:
        if set(object1.keys()) != set(object2.keys()):
            if config.show_difference:
                logger.info(
                    f"objects have different keys:\n"
                    f"object1 keys: {sorted(set(object1.keys()))}\n"
                    f"object2 keys: {sorted(set(object2.keys()))}"
                )
            return False
        return self._handle_next(object1=object1, object2=object2, config=config)
