r"""Implement equality testers for collection types.

This module provides equality testers for Python's built-in collection
types: sequences (list, tuple, deque) and mappings (dict). These testers
recursively compare nested structures using the equality tester
registry.
"""

from __future__ import annotations

__all__ = [
    "MappingEqualityTester",
    "OrderedMappingEqualityTester",
    "SequenceEqualityTester",
    "SetEqualityTester",
]

from collections.abc import Mapping, Sequence, Set as AbstractSet
from typing import TYPE_CHECKING, Any

from coola.equality.handler import (
    MappingSameKeyOrderHandler,
    MappingSameKeysHandler,
    MappingSameValuesHandler,
    SameLengthHandler,
    SameObjectHandler,
    SameTypeHandler,
    SequenceSameValuesHandler,
    SetSameValuesHandler,
    TrueHandler,
    create_chain,
)
from coola.equality.tester.chain import BaseChainEqualityTester

if TYPE_CHECKING:
    from coola.equality.handler import BaseEqualityHandler


class MappingEqualityTester(BaseChainEqualityTester[Mapping[Any, Any]]):
    r"""Implement a mapping equality tester.

    This tester handles dictionary-like objects (dict, Mapping ABC) by recursively
    comparing their keys and values. The handler chain:
    1. SameObjectHandler: Check for object identity
    2. SameTypeHandler: Verify same type
    3. SameLengthHandler: Check both mappings have same number of keys
    4. MappingSameKeysHandler: Verify both have the same keys
    5. MappingSameValuesHandler: Recursively compare values using registry
    6. TrueHandler: Return True if all checks pass

    The values are compared recursively, so nested dictionaries, lists, and
    other complex structures are handled correctly.

    Example:
        Basic mapping comparison:

        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import MappingEqualityTester
        >>> config = EqualityConfig()
        >>> tester = MappingEqualityTester()
        >>> tester.objects_are_equal({"a": 1, "b": 2}, {"a": 1, "b": 2}, config)
        True
        >>> tester.objects_are_equal({"a": 1, "b": 2}, {"a": 1, "b": 4}, config)
        False

        ```

        Nested mapping comparison:

        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import MappingEqualityTester
        >>> config = EqualityConfig()
        >>> tester = MappingEqualityTester()
        >>> tester.objects_are_equal(
        ...     {"a": {"x": 1}, "b": [1, 2]},
        ...     {"a": {"x": 1}, "b": [1, 2]},
        ...     config,
        ... )
        True

        ```
    """

    def _create_handler(self) -> BaseEqualityHandler:
        return create_chain(
            SameObjectHandler(),
            SameTypeHandler(),
            SameLengthHandler(),
            MappingSameKeysHandler(),
            MappingSameValuesHandler(),
            TrueHandler(),
        )


class OrderedMappingEqualityTester(BaseChainEqualityTester[Mapping[Any, Any]]):
    r"""Implement an equality tester for mappings whose order matters.

    This tester behaves like ``MappingEqualityTester``, and additionally
    requires the keys to be in the same order. It is registered for
    ``collections.OrderedDict``, for which Python's ``==`` is also
    order-sensitive. The handler chain:
    1. SameObjectHandler: Check for object identity
    2. SameTypeHandler: Verify same type
    3. SameLengthHandler: Check both mappings have same number of keys
    4. MappingSameKeysHandler: Verify both have the same keys
    5. MappingSameKeyOrderHandler: Verify the keys are in the same order
    6. MappingSameValuesHandler: Recursively compare values using registry
    7. TrueHandler: Return True if all checks pass

    Example:
        ```pycon
        >>> from collections import OrderedDict
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import OrderedMappingEqualityTester
        >>> config = EqualityConfig()
        >>> tester = OrderedMappingEqualityTester()
        >>> tester.objects_are_equal(OrderedDict(a=1, b=2), OrderedDict(a=1, b=2), config)
        True
        >>> tester.objects_are_equal(OrderedDict(a=1, b=2), OrderedDict(b=2, a=1), config)
        False

        ```
    """

    def _create_handler(self) -> BaseEqualityHandler:
        return create_chain(
            SameObjectHandler(),
            SameTypeHandler(),
            SameLengthHandler(),
            MappingSameKeysHandler(),
            MappingSameKeyOrderHandler(),
            MappingSameValuesHandler(),
            TrueHandler(),
        )


class SequenceEqualityTester(BaseChainEqualityTester[Sequence[Any]]):
    r"""Implement a sequence equality tester.

    This tester handles sequence types (list, tuple, deque, Sequence ABC) by
    recursively comparing their elements. The handler chain:
    1. SameObjectHandler: Check for object identity
    2. SameTypeHandler: Verify same type
    3. SameLengthHandler: Check both sequences have same length
    4. SequenceSameValuesHandler: Recursively compare elements using registry
    5. TrueHandler: Return True if all checks pass

    Elements are compared in order and recursively, so nested lists, dicts,
    and other complex structures are handled correctly.

    Example:
        Basic sequence comparison:

        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import SequenceEqualityTester
        >>> config = EqualityConfig()
        >>> tester = SequenceEqualityTester()
        >>> tester.objects_are_equal([1, 2, 3], [1, 2, 3], config)
        True
        >>> tester.objects_are_equal([1, 2, 3], [1, 2, 4], config)
        False

        ```

        Nested sequence comparison:

        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import SequenceEqualityTester
        >>> config = EqualityConfig()
        >>> tester = SequenceEqualityTester()
        >>> tester.objects_are_equal(
        ...     [[1, 2], {"a": 3}],
        ...     [[1, 2], {"a": 3}],
        ...     config,
        ... )
        True

        ```
    """

    def _create_handler(self) -> BaseEqualityHandler:
        return create_chain(
            SameObjectHandler(),
            SameTypeHandler(),
            SameLengthHandler(),
            SequenceSameValuesHandler(),
            TrueHandler(),
        )


class SetEqualityTester(BaseChainEqualityTester[AbstractSet[Any]]):
    r"""Implement a set equality tester.

    This tester handles ``set`` and ``frozenset``. Unlike ``==``, the
    elements are compared with the registry, so ``equal_nan`` and the
    tolerance apply to them. The handler chain:
    1. SameObjectHandler: Check for object identity
    2. SameTypeHandler: Verify same type
    3. SameLengthHandler: Check both sets have same length
    4. SetSameValuesHandler: Match each element of the first set with
       an element of the second set, using the registry
    5. TrueHandler: Return True if all checks pass

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import SetEqualityTester
        >>> config = EqualityConfig(equal_nan=True)
        >>> tester = SetEqualityTester()
        >>> tester.objects_are_equal({1.0, float("nan")}, {1.0, float("nan")}, config)
        True
        >>> tester.objects_are_equal({1.0, 2.0}, {1.0, 3.0}, config)
        False

        ```
    """

    def _create_handler(self) -> BaseEqualityHandler:
        return create_chain(
            SameObjectHandler(),
            SameTypeHandler(),
            SameLengthHandler(),
            SetSameValuesHandler(),
            TrueHandler(),
        )
