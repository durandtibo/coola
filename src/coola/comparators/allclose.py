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
    r"""Implements a default allclose operator.

    The ``==`` operator is used to test the equality between the objects
    because it is not possible to define an allclose operator for all
    objects.
    """

    def __eq__(self, other: Any) -> bool:
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
    r"""Implements an equality operator for mappings."""

    def __eq__(self, other: Any) -> bool:
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
    r"""Implements an allclose operator for scalar values."""

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__)

    def allclose(
        self,
        tester: BaseAllCloseTester,
        object1: bool | int | float,
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
        number_equal = math.isclose(object1, object2, rel_tol=rtol, abs_tol=atol)
        if show_difference and not number_equal:
            logger.info(f"The numbers are different: {object1} vs {object2}")
        return number_equal

    def clone(self) -> ScalarAllCloseOperator:
        return self.__class__()


class SequenceAllCloseOperator(BaseAllCloseOperator[Sequence]):
    r"""Implements an equality operator for sequences."""

    def __eq__(self, other: Any) -> bool:
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
    r"""Gets a default mapping between the types and the allclose
    operators.

    Returns:
        dict: The mapping between the types and the allclose
            operators.
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
