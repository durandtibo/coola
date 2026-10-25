r"""Implement handlers for set objects."""

from __future__ import annotations

__all__ = ["SetSameValuesHandler"]

from collections import Counter
from typing import TYPE_CHECKING, Any

from coola.equality.handler.base import BaseEqualityHandler
from coola.equality.handler.mixin import HandlerEqualityMixin
from coola.equality.handler.utils import check_recursion_depth, comparison_key, report_difference
from coola.utils.safe import safe_repr

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet

    from coola.equality.config import EqualityConfig


class SetSameValuesHandler(HandlerEqualityMixin, BaseEqualityHandler):
    r"""Check if the two sets have the same elements.

    The sets are compared with the hash of their elements, in linear
    time, so ``equal_nan`` applies to the elements, unlike the ``==`` of
    the sets (the sets ``{(float("nan"), 1)}`` and
    ``{(float("nan"), 1)}`` are equal with ``equal_nan=True``). The
    type of the elements is part of the comparison, so ``{1}`` and
    ``{True}`` are different. The tolerance is not applied: the
    elements are compared exactly, because close values have different
    hashes.

    This handler returns ``False`` if the two sets have a different
    length or if one element of ``actual`` has no match, otherwise it
    passes the inputs to the next handler.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import SetSameValuesHandler, TrueHandler
        >>> config = EqualityConfig(equal_nan=True)
        >>> handler = SetSameValuesHandler(next_handler=TrueHandler())
        >>> handler.handle({1.0, float("nan")}, {1.0, float("nan")}, config)
        True
        >>> handler.handle({1.0, 2.0}, {1.0, 3.0}, config)
        False

        ```
    """

    def handle(
        self,
        actual: AbstractSet[Any],
        expected: AbstractSet[Any],
        config: EqualityConfig,
    ) -> bool:
        with check_recursion_depth(config):
            if len(actual) != len(expected):
                if config.explaining:
                    report_difference(
                        config,
                        "sets have different lengths: %s vs %s",
                        f"{len(actual):,}",
                        f"{len(expected):,}",
                    )
                return False
            unmatched = self._find_unmatched(actual, expected, config)
            if unmatched is not _NO_ELEMENT:
                if config.explaining:
                    report_difference(
                        config,
                        "sets have different elements: %s of actual has no match in expected",
                        safe_repr(unmatched),
                        path=(unmatched,),
                    )
                return False
            return self._handle_next(actual, expected, config=config)

    @classmethod
    def _find_unmatched(
        cls, actual: AbstractSet[Any], expected: AbstractSet[Any], config: EqualityConfig
    ) -> Any:
        r"""Find an element of ``actual`` that has no match in
        ``expected``.

        Returns:
            The first element of ``actual`` that has no match, or
                ``_NO_ELEMENT`` if every element has one.
        """
        counts = Counter(comparison_key(value, config) for value in expected)
        for value in actual:
            key = comparison_key(value, config)
            if counts[key] == 0:
                return value
            counts[key] -= 1
        return _NO_ELEMENT


_NO_ELEMENT: Any = object()
