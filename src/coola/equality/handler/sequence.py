r"""Implement handlers for sequence objects."""

from __future__ import annotations

__all__ = ["SequenceSameValuesHandler", "is_exact_scalar_sequence"]

from typing import TYPE_CHECKING, Any

from coola.equality.handler.base import BaseEqualityHandler
from coola.equality.handler.format import format_sequence_difference
from coola.equality.handler.mixin import HandlerEqualityMixin
from coola.equality.handler.utils import (
    check_recursion_depth,
    report_difference,
    track_comparison,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from coola.equality.config import EqualityConfig

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
        with (
            check_recursion_depth(config),
            track_comparison(config, actual, expected) as in_progress,
        ):
            if in_progress:
                # Cycle: the comparison of this pair is already in progress.
                return True
            if len(actual) != len(expected):
                if config.show_difference:
                    report_difference(
                        config,
                        "sequences have different lengths: %s vs %s",
                        f"{len(actual):,}",
                        f"{len(expected):,}",
                    )
                return False
            # Fast path: skip the per-element dispatch. ``==`` ignores the type
            # (``False == 0``), so the types must match item by item. If the
            # sequences differ and there is no tolerance (the integers are compared
            # within a tolerance), the answer is ``False`` at once: the slow path is only
            # needed to locate the first difference when it is reported.
            if (
                is_exact_scalar_sequence(actual)
                and is_exact_scalar_sequence(expected)
                and list(map(type, actual)) == list(map(type, expected))
            ):
                if actual == expected:
                    return self._handle_next(actual, expected, config=config)
                if not config.show_difference and config.atol == 0.0 and config.rtol == 0.0:
                    return False
            for idx, (value1, value2) in enumerate(zip(actual, expected, strict=True)):
                if not config.registry.objects_are_equal(value1, value2, config):
                    if config.show_difference:
                        report_difference(
                            config,
                            format_sequence_difference(
                                actual,
                                expected,
                                different_index=idx,
                            ),
                            path=(idx,),
                        )
                    return False
            return self._handle_next(actual, expected, config=config)


def is_exact_scalar_sequence(seq: Sequence[Any]) -> bool:
    r"""Indicate if ``seq`` is a list or tuple containing only exact
    built-in scalars.

    The exact scalars are the ``int``, ``str``, ``bytes``, ``bool`` and
    ``None`` values: ``==`` is exact for them (no NaN, no tolerance), so
    a sequence of exact scalars can be compared at once. The type of
    ``seq`` and the types of its items are compared exactly (no
    subclass), so a subclass of ``list`` is not accepted, and neither is
    a ``float`` (it can be NaN or compared with a tolerance).

    Args:
        seq: The sequence to check.

    Returns:
        ``True`` if ``seq`` is a ``list`` or a ``tuple`` and all its
            items have an exact scalar type, otherwise ``False``.

    Example:
        ```pycon
        >>> from coola.equality.handler.sequence import is_exact_scalar_sequence
        >>> is_exact_scalar_sequence([1, "abc", None])
        True
        >>> is_exact_scalar_sequence([1, 2.0])
        False
        >>> is_exact_scalar_sequence([1, [2]])
        False

        ```
    """
    return type(seq) in (list, tuple) and set(map(type, seq)) <= _EXACT_SCALAR_TYPES
