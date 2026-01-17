r"""Implement handlers for sequence objects."""

from __future__ import annotations

__all__ = ["SequenceSameValuesHandler"]

import logging
from typing import TYPE_CHECKING, Any

from coola.equality.format import format_sequence_difference
from coola.equality.handler.base import BaseEqualityHandler
from coola.equality.handler.utils import check_recursion_depth, handlers_are_equal

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
        with check_recursion_depth(config):
            for idx, (value1, value2) in enumerate(zip(actual, expected)):
                # Add path element before recursing
                config.add_path_element(f"[index {idx}]")
                equal = config.registry.objects_are_equal(value1, value2, config)
                # Remove path element after recursing (whether equal or not)
                config.remove_last_path_element()
                if not equal:
                    return False
            return self._handle_next(actual, expected, config=config)

    def _show_difference(
        self, 
        actual: Sequence, 
        expected: Sequence, 
        index: int,
        config: EqualityConfig,
    ) -> None:
        # This method is no longer needed as we handle path in the recursive call
        pass
