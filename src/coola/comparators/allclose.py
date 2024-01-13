r"""Implement the ``allclose`` equality operators for the python native
types."""

from __future__ import annotations

__all__ = [
    "DefaultAllCloseOperator",
    "MappingAllCloseOperator",
    "ScalarAllCloseOperator",
    "SequenceAllCloseOperator",
    "get_mapping_allclose",
]

import logging
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Union

from coola.comparators.base import BaseAllCloseOperator

if TYPE_CHECKING:
    from coola.testers import BaseAllCloseTester

logger = logging.getLogger(__name__)


class DefaultAllCloseOperator(BaseAllCloseOperator[Any]):
    r"""Implement a default allclose operator.

    The ``==`` operator is used to test the equality between the objects
    because it is not possible to define an allclose operator for all
    objects.

    ```pycon
    >>> from coola.testers import AllCloseTester
    >>> from coola.comparators import DefaultAllCloseOperator
    >>> tester = AllCloseTester()
    >>> op = DefaultAllCloseOperator()
    >>> op.allclose(tester, 42, 42)
    True
    >>> DefaultAllCloseOperator().allclose(tester, "meow", "meov")
    False

    ```
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def allclose(
        self,
        tester: BaseAllCloseTester,
        object1: Any,
        object2: Any,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
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

    def clone(self) -> DefaultAllCloseOperator:
        return self.__class__()


class MappingAllCloseOperator(BaseAllCloseOperator[Mapping]):
    r"""Implement an equality operator for mappings.

    ```pycon
    >>> from coola.testers import AllCloseTester
    >>> from coola.comparators import MappingAllCloseOperator
    >>> tester = AllCloseTester()
    >>> op = MappingAllCloseOperator()
    >>> op.allclose(
    ...     tester,
    ...     {'key1': 42, 'key2': 1.2, "key3": "abc"},
    ...     {'key1': 42, 'key2': 1.2, "key3": "abc"}
    ... )
    True
    >>> MappingAllCloseOperator().allclose(
    ...     tester,
    ...     {'key1': 42, 'key2': 1.2, "key3": "abc"},
    ...     {'key1': 42, 'key2': 1.201, "key3": "abc"}
    ... )
    False
    >>> MappingAllCloseOperator().allclose(
    ...     tester,
    ...     {'key1': 42, 'key2': 1.2, "key3": "abc"},
    ...     {'key1': 42, 'key2': 1.201, "key3": "abc"},
    ...     atol=1e-2,
    ... )
    True

    ```
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def allclose(
        self,
        tester: BaseAllCloseTester,
        object1: Mapping,
        object2: Any,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
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
        keys1, keys2 = sorted(object1.keys()), sorted(object2.keys())
        if not tester.allclose(keys1, keys2, rtol, atol, equal_nan, show_difference):
            if show_difference:
                logger.info(
                    f"The mappings have different keys:\n"
                    f"keys of object1: {keys1}\n"
                    f"keys of object2: {keys2}"
                )
            return False
        for key1, key2 in zip(keys1, keys2):
            if not tester.allclose(
                object1[key1], object2[key2], rtol, atol, equal_nan, show_difference
            ):
                if show_difference:
                    logger.info(
                        f"The mappings have a different value for the key '{key1}':\n"
                        f"first mapping  = {object1}\n"
                        f"second mapping = {object2}"
                    )
                return False
        return True

    def clone(self) -> MappingAllCloseOperator:
        return self.__class__()


class ScalarAllCloseOperator(BaseAllCloseOperator[Union[bool, int, float]]):
    r"""Implement an allclose operator for scalar values.

    ```pycon
    >>> from coola.testers import AllCloseTester
    >>> from coola.comparators import ScalarAllCloseOperator
    >>> tester = AllCloseTester()
    >>> op = ScalarAllCloseOperator()
    >>> op.allclose(tester, 42, 42)
    True
    >>> op.allclose(tester, 42.0, 42.001)
    False
    >>> op.allclose(tester, 42.0, 42.001, atol=1e-2)
    True

    ```
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def allclose(
        self,
        tester: BaseAllCloseTester,
        object1: bool | float,
        object2: Any,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
        show_difference: bool = False,
    ) -> bool:
        if object1 is object2:
            return True
        if type(object1) is not type(object2):
            if show_difference:
                logger.info(f"Objects have different types: {type(object1)} vs {type(object2)}")
            return False
        if equal_nan and math.isnan(object1) and math.isnan(object2):
            return True
        number_equal = math.isclose(object1, object2, rel_tol=rtol, abs_tol=atol)
        if show_difference and not number_equal:
            logger.info(f"The numbers are different: {object1} vs {object2}")
        return number_equal

    def clone(self) -> ScalarAllCloseOperator:
        return self.__class__()


class SequenceAllCloseOperator(BaseAllCloseOperator[Sequence]):
    r"""Implement an equality operator for sequences.

    ```pycon
    >>> from coola.testers import AllCloseTester
    >>> from coola.comparators import SequenceAllCloseOperator
    >>> tester = AllCloseTester()
    >>> op = SequenceAllCloseOperator()
    >>> op.allclose(tester, [42, 1.2, "abc"], [42, 1.2, "abc"])
    True
    >>> op.allclose(tester, [42, 1.2, "abc"], [42, 1.201, "abc"])
    False
    >>> op.allclose(tester, [42, 1.2, "abc"], [42, 1.201, "abc"], atol=1e-2)
    True

    ```
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def allclose(
        self,
        tester: BaseAllCloseTester,
        object1: Sequence,
        object2: Any,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
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
            if not tester.allclose(value1, value2, rtol, atol, equal_nan, show_difference):
                if show_difference:
                    logger.info(
                        f"The sequences have at least one different value:\n"
                        f"first sequence  = {object1}\n"
                        f"second sequence = {object2}"
                    )
                return False
        return True

    def clone(self) -> SequenceAllCloseOperator:
        return self.__class__()


def get_mapping_allclose() -> dict[type[object], BaseAllCloseOperator]:
    r"""Get a default mapping between the types and the allclose
    operators.

    Returns:
        The mapping between the types and the allclose operators.

    ```pycon
    >>> from coola.comparators.allclose import get_mapping_allclose
    >>> get_mapping_allclose()
    {<class 'collections.abc.Mapping'>: MappingAllCloseOperator(),
     <class 'collections.abc.Sequence'>: SequenceAllCloseOperator(),
     <class 'bool'>: ScalarAllCloseOperator(),
     <class 'dict'>: MappingAllCloseOperator(),
     <class 'float'>: ScalarAllCloseOperator(),
     <class 'int'>: ScalarAllCloseOperator(),
     <class 'list'>: SequenceAllCloseOperator(),
     <class 'object'>: DefaultAllCloseOperator(),
     <class 'tuple'>: SequenceAllCloseOperator()}

    ```
    """
    return {
        Mapping: MappingAllCloseOperator(),
        Sequence: SequenceAllCloseOperator(),
        bool: ScalarAllCloseOperator(),
        dict: MappingAllCloseOperator(),
        float: ScalarAllCloseOperator(),
        int: ScalarAllCloseOperator(),
        list: SequenceAllCloseOperator(),
        object: DefaultAllCloseOperator(),
        tuple: SequenceAllCloseOperator(),
    }