r"""Implement equality testers for collection types.

This module provides equality testers for Python's built-in collection
types: sequences (list, tuple, deque), mappings (dict) and sets. The
testers return the child pairs that must all be equal, and the engine
compares them.
"""

from __future__ import annotations

__all__ = [
    "MappingEqualityTester",
    "OrderedMappingEqualityTester",
    "SequenceEqualityTester",
    "SetEqualityTester",
]

from collections import Counter
from itertools import count
from typing import TYPE_CHECKING, Any

from coola.equality._helpers.format import format_mapping_difference
from coola.equality._helpers.utils import comparison_key, is_exact_scalar_sequence
from coola.equality.engine import Element
from coola.equality.tester.base import BaseEqualityTester
from coola.utils.safe import safe_repr

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence, Set as AbstractSet

    from coola.equality.engine import Engine

_NO_ELEMENT: Any = object()
_PLAIN_KEY_TYPES: frozenset[type] = frozenset({str, int})


class MappingEqualityTester(BaseEqualityTester["Mapping[Any, Any]"]):
    r"""Implement a mapping equality tester.

    This tester handles dictionary-like objects (dict, Mapping ABC). The
    two mappings must have the same keys, and the values of the same
    key are compared in the order of ``actual``. The keys are compared
    with their type, like the elements of sets: ``{1: "a"}`` and
    ``{True: "a"}`` are not the same mapping. With ``equal_nan=True``,
    two NaN keys are the same key.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.engine import Engine
        >>> from coola.equality.tester import get_default_registry
        >>> registry = get_default_registry()
        >>> engine = Engine(EqualityConfig())
        >>> engine.run({"a": 1, "b": 2}, {"a": 1, "b": 2})
        True
        >>> engine.run({"a": 1, "b": 2}, {"a": 1, "b": 4})
        False

        ```
    """

    def objects_are_equal(
        self, actual: Mapping[Any, Any], expected: Mapping[Any, Any], ctx: Engine
    ) -> bool | Iterator[tuple[object, object, object]]:
        if (
            actual.keys() == expected.keys()
            and set(map(type, actual)) <= _PLAIN_KEY_TYPES
            and set(map(type, expected)) <= _PLAIN_KEY_TYPES
        ):
            # Fast path: the keys are strings or integers, which are equal if they are equal
            # as keys (no NaN, no ``1`` against ``True``).
            return zip(
                map(actual.__getitem__, actual),
                map(expected.__getitem__, actual),
                actual,
                strict=True,
            )
        keys_a = {comparison_key(key, ctx.config): key for key in actual}
        keys_e = {comparison_key(key, ctx.config): key for key in expected}
        if keys_a.keys() != keys_e.keys():
            return ctx.fail(
                lambda: format_mapping_difference(
                    missing_keys={keys_a[k] for k in keys_a.keys() - keys_e.keys()},
                    additional_keys={keys_e[k] for k in keys_e.keys() - keys_a.keys()},
                )
            )
        return ((actual[keys_a[k]], expected[keys_e[k]], keys_a[k]) for k in keys_a)


class OrderedMappingEqualityTester(MappingEqualityTester):
    r"""Implement an ordered mapping equality tester.

    This tester behaves like ``MappingEqualityTester``, and additionally
    requires the keys to be in the same order. It is registered for
    ``collections.OrderedDict``, for which Python's ``==`` is also
    order-sensitive.

    Example:
        ```pycon
        >>> from collections import OrderedDict
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.engine import Engine
        >>> from coola.equality.tester import get_default_registry
        >>> engine = Engine(EqualityConfig())
        >>> engine.run(OrderedDict(a=1, b=2), OrderedDict(a=1, b=2))
        True
        >>> engine.run(OrderedDict(a=1, b=2), OrderedDict(b=2, a=1))
        False

        ```
    """

    def objects_are_equal(
        self, actual: Mapping[Any, Any], expected: Mapping[Any, Any], ctx: Engine
    ) -> bool | Iterator[tuple[object, object, object]]:
        result = super().objects_are_equal(actual, expected, ctx)
        if result is False:
            return False
        actual_keys, expected_keys = list(actual), list(expected)
        if [comparison_key(key, ctx.config) for key in actual_keys] != [
            comparison_key(key, ctx.config) for key in expected_keys
        ]:
            return ctx.fail(
                lambda: format_mapping_difference(different_key_order=(actual_keys, expected_keys))
            )
        return result


class SequenceEqualityTester(BaseEqualityTester["Sequence[Any]"]):
    r"""Implement a sequence equality tester.

    This tester handles sequence types (list, tuple, deque, Sequence
    ABC). The two sequences must have the same length, and the items
    are compared in order. A sequence of exact scalars (``int``,
    ``str``, ``bytes``, ``bool`` and ``None``) is compared at once.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.engine import Engine
        >>> from coola.equality.tester import get_default_registry
        >>> engine = Engine(EqualityConfig())
        >>> engine.run([1, 2, 3], [1, 2, 3])
        True
        >>> engine.run([1, 2, 3], [1, 2, 4])
        False
        >>> engine.run([1, 2, 3], [1, 2])
        False

        ```
    """

    def objects_are_equal(
        self, actual: Sequence[Any], expected: Sequence[Any], ctx: Engine
    ) -> bool | Iterator[tuple[object, object, object]]:
        if len(actual) != len(expected):
            return ctx.fail(
                lambda: f"sequences have different lengths: {len(actual):,} vs {len(expected):,}"
            )
        # Fast path: skip the per-element dispatch. ``==`` ignores the type
        # (``False == 0``), so the types must match item by item. If the sequences differ and
        # there is no tolerance (the integers are compared within a tolerance), the answer is
        # ``False`` at once: the slow path is only needed to locate the first difference when
        # it is reported.
        if (
            is_exact_scalar_sequence(actual)
            and is_exact_scalar_sequence(expected)
            and list(map(type, actual)) == list(map(type, expected))
        ):
            if actual == expected:
                return True
            if not ctx.explaining and ctx.atol == 0.0 and ctx.rtol == 0.0:
                return False
        return zip(actual, expected, count(), strict=False)


class SetEqualityTester(BaseEqualityTester["AbstractSet[Any]"]):
    r"""Implement a set equality tester.

    This tester handles ``set`` and ``frozenset``. Unlike ``==``, the
    elements are matched by hash and type, so ``equal_nan`` applies to
    them (the tolerance does not). The location of a difference is the
    element of ``actual`` that has no match in ``expected``.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.engine import Engine
        >>> from coola.equality.tester import get_default_registry
        >>> engine = Engine(EqualityConfig())
        >>> engine.run({1, 2, 3}, {1, 2, 3})
        True
        >>> engine.run({1, 2, 3}, {1, 2, 4})
        False

        ```
    """

    def objects_are_equal(
        self, actual: AbstractSet[Any], expected: AbstractSet[Any], ctx: Engine
    ) -> bool:
        if len(actual) != len(expected):
            return ctx.fail(
                lambda: f"sets have different lengths: {len(actual):,} vs {len(expected):,}"
            )
        unmatched = self._find_unmatched(actual, expected, ctx)
        if unmatched is not _NO_ELEMENT:
            return ctx.fail(
                lambda: (
                    f"sets have different elements: {safe_repr(unmatched)} of actual "
                    "has no match in expected"
                ),
                Element(unmatched),
            )
        return True

    @classmethod
    def _find_unmatched(
        cls, actual: AbstractSet[Any], expected: AbstractSet[Any], ctx: Engine
    ) -> Any:
        r"""Find an element of ``actual`` that has no match in
        ``expected``.

        Args:
            actual: The first set.
            expected: The second set.
            ctx: The ``Engine`` of the comparison, which gives the
                comparison keys.

        Returns:
            The first element of ``actual`` that has no match, or
                ``_NO_ELEMENT`` if every element has one.
        """
        counts = Counter(comparison_key(value, ctx.config) for value in expected)
        for value in actual:
            key = comparison_key(value, ctx.config)
            if counts[key] == 0:
                return value
            counts[key] -= 1
        return _NO_ELEMENT
