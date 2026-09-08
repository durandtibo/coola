r"""Implement equality testers for collection types.

This module provides equality testers for Python's built-in collection
types: sequences (list, tuple, deque) and mappings (dict). These testers
recursively compare nested structures using the equality tester
registry.
"""

from __future__ import annotations

__all__ = ["MappingEqualityTester", "SequenceEqualityTester"]

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from coola.display import InlineDisplayMixin
from coola.equality.handler import (
    MappingSameKeysHandler,
    MappingSameValuesHandler,
    SameLengthHandler,
    SameObjectHandler,
    SameTypeHandler,
    SequenceSameValuesHandler,
    TrueHandler,
    create_chain,
)
from coola.equality.tester.base import BaseEqualityTester

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig


class MappingEqualityTester(InlineDisplayMixin, BaseEqualityTester[Mapping[Any, Any]]):
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

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {}

    def __init__(self) -> None:
        self._handler = create_chain(
            SameObjectHandler(),
            SameTypeHandler(),
            SameLengthHandler(),
            MappingSameKeysHandler(),
            MappingSameValuesHandler(),
            TrueHandler(),
        )


    def equal(self, other: object) -> bool:
        return type(other) is type(self)

    def objects_are_equal(
        self,
        actual: Mapping[Any, Any],
        expected: object,
        config: EqualityConfig,
    ) -> bool:
        return self._handler.handle(actual, expected, config=config)


class SequenceEqualityTester(InlineDisplayMixin, BaseEqualityTester[Sequence[Any]]):
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

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {}

    def __init__(self) -> None:
        self._handler = create_chain(
            SameObjectHandler(),
            SameTypeHandler(),
            SameLengthHandler(),
            SequenceSameValuesHandler(),
            TrueHandler(),
        )


    def equal(self, other: object) -> bool:
        return type(other) is type(self)

    def objects_are_equal(
        self,
        actual: Sequence[Any],
        expected: object,
        config: EqualityConfig,
    ) -> bool:
        return self._handler.handle(actual, expected, config=config)
