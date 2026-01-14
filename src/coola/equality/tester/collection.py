r"""Implement equality testers for collection types.

This module provides equality testers for Python's built-in collection types:
sequences (list, tuple, deque) and mappings (dict). These testers recursively
compare nested structures using the equality tester registry.
"""

from __future__ import annotations

__all__ = ["MappingEqualityTester", "SequenceEqualityTester"]

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from coola.equality.handler import (
    MappingSameKeysHandler,
    MappingSameValuesHandler,
    SameLengthHandler,
    SameObjectHandler,
    SameTypeHandler,
    SequenceSameValuesHandler,
    TrueHandler,
)
from coola.equality.tester.base import BaseEqualityTester

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig


class MappingEqualityTester(BaseEqualityTester[Mapping[Any, Any]]):
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

    def __init__(self) -> None:
        """Initialize the mapping equality tester with its handler chain.

        The handler chain performs checks in this order:
        1. SameObjectHandler: Quick check for object identity
        2. SameTypeHandler: Verify both objects have the same type
        3. SameLengthHandler: Check both mappings have the same number of items
        4. MappingSameKeysHandler: Verify both have the exact same keys
        5. MappingSameValuesHandler: Recursively compare values for each key
        6. TrueHandler: Return True if all previous checks passed
        """
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(SameLengthHandler()).chain(
            MappingSameKeysHandler()
        ).chain(MappingSameValuesHandler()).chain(TrueHandler())

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def equal(self, other: object) -> bool:
        return type(other) is type(self)

    def objects_are_equal(
        self,
        actual: Mapping[Any, Any],
        expected: object,
        config: EqualityConfig,
    ) -> bool:
        return self._handler.handle(actual, expected, config=config)


class SequenceEqualityTester(BaseEqualityTester[Sequence[Any]]):
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

    def __init__(self) -> None:
        """Initialize the sequence equality tester with its handler chain.

        The handler chain performs checks in this order:
        1. SameObjectHandler: Quick check for object identity
        2. SameTypeHandler: Verify both objects have the same type
        3. SameLengthHandler: Check both sequences have the same length
        4. SequenceSameValuesHandler: Recursively compare elements in order
        5. TrueHandler: Return True if all previous checks passed
        """
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(SameLengthHandler()).chain(
            SequenceSameValuesHandler()
        ).chain(TrueHandler())

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def equal(self, other: object) -> bool:
        return type(other) is type(self)

    def objects_are_equal(
        self,
        actual: Sequence[Any],
        expected: object,
        config: EqualityConfig,
    ) -> bool:
        return self._handler.handle(actual, expected, config=config)
