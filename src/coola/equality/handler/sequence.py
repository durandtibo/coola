r"""Implement handlers for sequence objects."""

from __future__ import annotations

__all__ = ["SequenceSameValuesHandler"]

import logging
from typing import TYPE_CHECKING, Any

from coola.equality.handler.base import BaseEqualityHandler
from coola.equality.handler.format import format_sequence_difference
from coola.equality.handler.mixin import HandlerEqualityMixin
from coola.equality.handler.utils import check_recursion_depth, report_difference

if TYPE_CHECKING:
    from collections.abc import Sequence

    from coola.equality.config import EqualityConfig

logger: logging.Logger = logging.getLogger(__name__)

# Types for which ``==`` is exact and agrees with ``objects_are_equal``
# (no NaN, no tolerance), so whole sequences can be compared at once.
_EXACT_SCALAR_TYPES: frozenset[type] = frozenset({int, str, bytes, bool, type(None)})


class SequenceSameValuesHandler(HandlerEqualityMixin, BaseEqualityHandler):
    r"""Check if the two sequences have the same values.

    This handler returns ``False`` if the two sequences have at least
    one different value or a different length, otherwise it passes the
    inputs to the next handler. The length check is done by this
    handler as defense-in-depth, so it returns a correct result even
    when used standalone or without a preceding ``SameLengthHandler``
    in the chain.

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
                    report_difference(
                        logger,
                        config,
                        "sequences have different lengths: %s vs %s",
                        f"{len(actual):,}",
                        f"{len(expected):,}",
                    )
                return False
            # Fast path: skip the per-element dispatch. On mismatch, fall
            # through to the slow path to locate the first difference.
            if (
                _is_exact_scalar_sequence(actual)
                and _is_exact_scalar_sequence(expected)
                and actual == expected
            ):
                return self._handle_next(actual, expected, config=config)
            for idx, (value1, value2) in enumerate(zip(actual, expected, strict=True)):
                if not config.registry.objects_are_equal(value1, value2, config):
                    if config.show_difference:
                        report_difference(
                            logger,
                            config,
                            format_sequence_difference(
                                actual,
                                expected,
                                different_index=idx,
                            ),
                            extra={"coola_path": (idx,), "coola_path_kind": "index"},
                        )
                    return False
            return self._handle_next(actual, expected, config=config)


def _is_exact_scalar_sequence(seq: Sequence[Any]) -> bool:
    r"""Indicate if ``seq`` is a list or tuple containing only exact
    built-in scalars."""
    return type(seq) in {list, tuple} and set(map(type, seq)) <= _EXACT_SCALAR_TYPES
