r"""Implement some handlers for sequence objects."""

from __future__ import annotations

__all__ = ["SequenceSameValueHandler"]

import logging
from typing import TYPE_CHECKING

from coola.equality.handlers.base import AbstractEqualityHandler

if TYPE_CHECKING:
    from collections.abc import Sequence

    from coola.equality.config import EqualityConfig

logger = logging.getLogger(__name__)


class SequenceSameValueHandler(AbstractEqualityHandler):
    r"""Check if the two sequences have the same values.

    This handler returns ``False`` if the two sequences have at least
    one different value, otherwise it passes the inputs to the next
    handler. If the sequences have different length, this handler
    checks only the values of the shortest sequence.

    Example usage:

    ```pycon
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.handlers import SequenceSameValueHandler, TrueHandler
    >>> from coola.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> handler = SequenceSameValueHandler(next_handler=TrueHandler())
    >>> handler.handle([1, 2, 3], [1, 2, 3], config)
    True
    >>> handler.handle([1, 2, 3], [1, 2, 4], config)
    False

    ```
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def handle(
        self,
        object1: Sequence,
        object2: Sequence,
        config: EqualityConfig,
    ) -> bool | None:
        for value1, value2 in zip(object1, object2):
            if not config.tester.equal(value1, value2, config.show_difference):
                if config.show_difference:
                    logger.info(
                        f"sequences have at least one different value:\n"
                        f"first sequence : {object1}\n"
                        f"second sequence: {object2}"
                    )
                return False
        return self._handle_next(object1=object1, object2=object2, config=config)
