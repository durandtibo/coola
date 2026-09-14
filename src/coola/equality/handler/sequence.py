r"""Implement handlers for sequence objects."""

from __future__ import annotations

__all__ = ["SequenceSameValuesHandler"]

import logging
from typing import TYPE_CHECKING, Any

from coola.equality.handler.base import BaseEqualityHandler
from coola.equality.handler.format import format_sequence_difference
from coola.equality.handler.mixin import HandlerEqualityMixin
from coola.equality.handler.utils import check_recursion_depth

if TYPE_CHECKING:
    from collections.abc import Sequence

    from coola.equality.config import EqualityConfig

logger: logging.Logger = logging.getLogger(__name__)


class SequenceSameValuesHandler(HandlerEqualityMixin, BaseEqualityHandler):
    r"""Check if the two sequences have the same values.

    This handler returns ``False`` if the two sequences have at least
    one different value or a different length, otherwise it passes the
    inputs to the next handler. The length check is done by this
    handler as defense-in-depth, so it returns a correct result even
    when used standalone or without a preceding ``SameLengthHandler``
    in the chain.

    Within a single ``handle`` call, the result of comparing a given
    pair of items is cached by object identity (``id()``), so repeated
    or interned values that appear at several indices are compared only
    once, avoiding redundant recursive ``objects_are_equal`` calls for
    highly repetitive sequences.

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
        >>> handler.handle([1, 2, 3], [1, 2], config)
        False

        ```
    """

    def handle(
        self,
        actual: Sequence[Any],
        expected: Sequence[Any],
        config: EqualityConfig,
    ) -> bool:
        with check_recursion_depth(config):
            if len(actual) != len(expected):
                if config.show_difference:
                    logger.info(
                        f"sequences have different lengths: {len(actual):,} vs {len(expected):,}"
                    )
                return False
            cache: dict[tuple[int, int], bool] = {}
            for idx, (value1, value2) in enumerate(zip(actual, expected)):
                key = (id(value1), id(value2))
                are_equal = cache.get(key)
                if are_equal is None:
                    are_equal = config.registry.objects_are_equal(value1, value2, config)
                    cache[key] = are_equal
                if not are_equal:
                    if config.show_difference:
                        logger.info(
                            format_sequence_difference(
                                actual,
                                expected,
                                different_index=idx,
                            )
                        )
                    return False
            return self._handle_next(actual, expected, config=config)
