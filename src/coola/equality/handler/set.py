r"""Implement handlers for set objects."""

from __future__ import annotations

__all__ = ["SetSameValuesHandler"]

import logging
from typing import TYPE_CHECKING, Any

from coola.equality.handler.base import BaseEqualityHandler
from coola.equality.handler.mixin import HandlerEqualityMixin
from coola.equality.handler.utils import check_recursion_depth, report_difference
from coola.utils.safe import safe_repr

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet

    from coola.equality.config import EqualityConfig

logger: logging.Logger = logging.getLogger(__name__)

# Types for which ``==`` is exact and agrees with ``objects_are_equal`` (no NaN, no tolerance).
_EXACT_SCALAR_TYPES: frozenset[type] = frozenset({int, str, bytes, bool, type(None)})


class SetSameValuesHandler(HandlerEqualityMixin, BaseEqualityHandler):
    r"""Check if the two sets have the same elements.

    The elements are compared with the registry, so ``equal_nan`` and
    the tolerance apply to them, unlike the ``==`` of the sets (the
    sets ``{(float("nan"), 1)}`` and ``{(float("nan"), 1)}`` are equal
    with ``equal_nan=True``, and ``{1.0}`` and ``{1.0 + 1e-9}`` are
    close). Each element of ``actual`` is matched with one element of
    ``expected`` of the same type that is not matched yet. With a
    tolerance, an element is matched with the first element of
    ``expected`` that is within the tolerance. The elements of
    ``expected`` are bucketed by type, and a set of exact scalars
    (``int``, ``str``, ``bytes``, ``bool`` and ``None``) is compared at
    once when there is no tolerance. The matching of the other elements
    is quadratic in the worst case.

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
                if config.show_difference:
                    report_difference(
                        logger,
                        config,
                        "sets have different lengths: %s vs %s",
                        f"{len(actual):,}",
                        f"{len(expected):,}",
                    )
                return False
            if not self._are_exact_and_equal(actual, expected, config):
                unmatched = self._find_unmatched(actual, expected, config)
                if unmatched is not _NO_ELEMENT:
                    if config.show_difference:
                        report_difference(
                            logger,
                            config,
                            "sets have different elements: %s of actual has no match in expected",
                            safe_repr(unmatched),
                            extra={"coola_path": (unmatched,)},
                        )
                    return False
            return self._handle_next(actual, expected, config=config)

    @staticmethod
    def _are_exact_and_equal(
        actual: AbstractSet[Any], expected: AbstractSet[Any], config: EqualityConfig
    ) -> bool:
        r"""Indicate if the sets contain only exact scalars and are
        equal.

        ``False`` does not mean that the sets are different: the
        elements must be matched one by one.
        """
        if config.atol != 0.0 or config.rtol != 0.0:
            return False
        if not (
            set(map(type, actual)) <= _EXACT_SCALAR_TYPES
            and set(map(type, expected)) <= _EXACT_SCALAR_TYPES
        ):
            return False
        # ``==`` ignores the type of the elements (``{1} == {True}``): compare the pairs.
        return {(type(x), x) for x in actual} == {(type(x), x) for x in expected}

    @staticmethod
    def _find_unmatched(
        actual: AbstractSet[Any], expected: AbstractSet[Any], config: EqualityConfig
    ) -> Any:
        r"""Find an element of ``actual`` that has no match in
        ``expected``.

        Returns:
            The first element of ``actual`` that has no match, or
                ``_NO_ELEMENT`` if every element has one.
        """
        candidates: dict[type, list[Any]] = {}
        for value in expected:
            candidates.setdefault(type(value), []).append(value)
        # The differences between the candidates are not reported: only the element
        # that has no match is, by the caller.
        show_difference, config.show_difference = config.show_difference, False
        try:
            for value in actual:
                bucket = candidates.get(type(value), [])
                for index, other in enumerate(bucket):
                    if config.registry.objects_are_equal(value, other, config):
                        del bucket[index]
                        break
                else:
                    return value
        finally:
            config.show_difference = show_difference
        return _NO_ELEMENT


_NO_ELEMENT: Any = object()
