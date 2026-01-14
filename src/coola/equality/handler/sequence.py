r"""Implement some handler for sequence objects."""

from __future__ import annotations

__all__ = ["SequenceSameValuesHandler"]

import logging
from typing import TYPE_CHECKING, Any

from coola.equality.handler.base import BaseEqualityHandler
from coola.equality.handler.utils import handlers_are_equal

if TYPE_CHECKING:
    from collections.abc import Sequence

    from coola.equality.config import EqualityConfig

logger: logging.Logger = logging.getLogger(__name__)


class SequenceSameValuesHandler(BaseEqualityHandler):
    r"""Check if the two sequences have the same values.

    This handler returns ``False`` if the two sequences have at least
    one different value, otherwise it passes the inputs to the next
    handler. If the sequences have different length, this handler
    checks only the values of the shortest sequence.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import SequenceSameValuesHandler, TrueHandler
        >>> config = EqualityConfig()
        >>> handler = SequenceSameValuesHandler(next_handler=TrueHandler())
        >>> handler.handle([1, 2, 3], [1, 2, 3], config)
        True
        >>> handler.handle([1, 2, 3], [1, 2, 4], config)
        False

        ```
    """

    def equal(self, other: object) -> bool:
        if type(other) is not type(self):
            return False
        return handlers_are_equal(self.next_handler, other.next_handler)

    def handle(
        self,
        actual: Sequence[Any],
        expected: Sequence[Any],
        config: EqualityConfig,
    ) -> bool:
        for value1, value2 in zip(actual, expected):
            if not config.registry.objects_are_equal(value1, value2, config):
                self._show_difference(actual, expected, config=config)
                return False
        return self._handle_next(actual, expected, config=config)

    def _show_difference(
        self, actual: Sequence, expected: Sequence, config: EqualityConfig
    ) -> None:
        if config.show_difference:
            logger.info(
                f"sequences have at least one different value:\n"
                f"first sequence : {actual}\n"
                f"second sequence: {expected}"
            )
