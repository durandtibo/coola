from __future__ import annotations

__all__ = [
    "DefaultEqualityOperator",
    "MappingEqualityOperator",
    "SequenceEqualityOperator",
    "get_mapping_equality",
]

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from coola.comparators.base import BaseEqualityOperator

if TYPE_CHECKING:
    from coola.testers import BaseEqualityTester

logger = logging.getLogger(__name__)


class DefaultEqualityOperator(BaseEqualityOperator[Any]):
    r"""Implements a default equality operator.

    The ``==`` operator is used to test the equality between the
    objects.

    Example usage:

    ```pycon
    >>> from coola.comparators import DefaultEqualityOperator
    >>> from coola.testers import EqualityTester
    >>> tester = EqualityTester()
    >>> op = DefaultEqualityOperator()
    >>> op.equal(tester, 42, 42)
    True
    >>> op.equal(tester, "meow", "meov")
    False

    ```
    """

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> DefaultEqualityOperator:
        return self.__class__()

    def equal(
        self,
        tester: BaseEqualityTester,
        object1: Any,
        object2: Any,
        show_difference: bool = False,
    ) -> bool:
        if object1 is object2:
            return True
        if type(object1) is not type(object2):
            if show_difference:
                logger.info(f"Objects have different types: {type(object1)} vs {type(object2)}")
            return False
        object_equal = object1 == object2
        if show_difference and not object_equal:
            logger.info(f"Objects are different:\nobject1={object1}\nobject2={object2}")
        return object_equal


class MappingEqualityOperator(BaseEqualityOperator[Mapping]):
    r"""Implements an equality operator for mappings.

    Example usage:

    ```pycon
    >>> from coola.comparators import MappingEqualityOperator
    >>> from coola.testers import EqualityTester
    >>> tester = EqualityTester()
    >>> op = MappingEqualityOperator()
    >>> op.equal(
    ...     tester,
    ...     {"key1": 42, "key2": 1.2, "key3": "abc"},
    ...     {"key1": 42, "key2": 1.2, "key3": "abc"},
    ... )
    True

    ```
    """

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> MappingEqualityOperator:
        return self.__class__()

    def equal(
        self,
        tester: BaseEqualityTester,
        object1: Mapping,
        object2: Any,
        show_difference: bool = False,
    ) -> bool:
        if object1 is object2:
            return True
        if type(object1) is not type(object2):
            if show_difference:
                logger.info(
                    f"The mappings have different types: {type(object1)} vs {type(object2)}"
                )
            return False
        if len(object1) != len(object2):
            if show_difference:
                logger.info(f"The mappings have different sizes: {len(object1)} vs {len(object2)}")
            return False
        if set(object1.keys()) != set(object2.keys()):
            if show_difference:
                logger.info(
                    f"The mappings have different keys:\n"
                    f"keys of object1: {sorted(set(object1.keys()))}\n"
                    f"keys of object2: {sorted(set(object2.keys()))}"
                )
            return False
        for key in object1.keys():
            if not tester.equal(object1[key], object2[key], show_difference):
                if show_difference:
                    logger.info(
                        f"The mappings have a different value for the key '{key}':\n"
                        f"first mapping  = {object1}\n"
                        f"second mapping = {object2}"
                    )
                return False
        return True


class SequenceEqualityOperator(BaseEqualityOperator[Sequence]):
    r"""Implements an equality operator for sequences.

    Example usage:

    ```pycon
    >>> from coola.comparators import SequenceEqualityOperator
    >>> from coola.testers import EqualityTester
    >>> tester = EqualityTester()
    >>> op = SequenceEqualityOperator()
    >>> op.equal(tester, [42, 1.2, "abc"], [42, 1.2, "abc"])
    True

    ```
    """

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> SequenceEqualityOperator:
        return self.__class__()

    def equal(
        self,
        tester: BaseEqualityTester,
        object1: Sequence,
        object2: Any,
        show_difference: bool = False,
    ) -> bool:
        if object1 is object2:
            return True
        if type(object1) is not type(object2):
            if show_difference:
                logger.info(
                    f"The sequences have different types: {type(object1)} vs {type(object2)}"
                )
            return False
        if len(object1) != len(object2):
            if show_difference:
                logger.info(
                    f"The sequences have different sizes: {len(object1):,} vs {len(object2):,}"
                )
            return False
        for value1, value2 in zip(object1, object2):
            if not tester.equal(value1, value2, show_difference):
                if show_difference:
                    logger.info(
                        f"The sequences have at least one different value:\n"
                        f"first sequence  = {object1}\n"
                        f"second sequence = {object2}"
                    )
                return False
        return True


def get_mapping_equality() -> dict[type[object], BaseEqualityOperator]:
    r"""Gets a default mapping between the types and the equality
    operators.

    Returns:
        dict: The mapping between the types and the equality
            operators.

    Example usage:

    ```pycon
    >>> from coola.comparators.equality import get_mapping_equality
    >>> get_mapping_equality()
    {<class 'collections.abc.Mapping'>: MappingEqualityOperator(),
     <class 'collections.abc.Sequence'>: SequenceEqualityOperator(),
     <class 'dict'>: MappingEqualityOperator(),
     <class 'list'>: SequenceEqualityOperator(),
     <class 'object'>: DefaultEqualityOperator(),
     <class 'tuple'>: SequenceEqualityOperator()}

    ```
    """
    return {
        Mapping: MappingEqualityOperator(),
        Sequence: SequenceEqualityOperator(),
        dict: MappingEqualityOperator(),
        list: SequenceEqualityOperator(),
        object: DefaultEqualityOperator(),
        tuple: SequenceEqualityOperator(),
    }
