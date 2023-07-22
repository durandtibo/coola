from __future__ import annotations

__all__ = [
    "NDArrayAllCloseOperator",
    "NDArrayEqualityOperator",
    "get_mapping_allclose",
    "get_mapping_equality",
]

import logging
from typing import TYPE_CHECKING, Any

from coola.comparators.base import BaseAllCloseOperator, BaseEqualityOperator
from coola.utils.imports import check_numpy, is_numpy_available

if TYPE_CHECKING:
    from coola.testers import BaseAllCloseTester, BaseEqualityTester

if is_numpy_available():
    from numpy import allclose, array_equal, ndarray
else:
    ndarray, array_equal, allclose = None, None, None  # pragma: no cover

logger = logging.getLogger(__name__)


class NDArrayAllCloseOperator(BaseAllCloseOperator[ndarray]):
    r"""Implements an allclose operator for ``numpy.ndarray``.

    Args:
        check_dtype (bool, optional): If ``True``, the data type of
            the arrays are checked, otherwise the data types are
            ignored. Default: ``True``
    """

    def __init__(self, check_dtype: bool = True) -> None:
        check_numpy()
        self._check_dtype = bool(check_dtype)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._check_dtype == other._check_dtype

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(check_dtype={self._check_dtype})"

    def allclose(
        self,
        tester: BaseAllCloseTester,
        object1: ndarray,
        object2: Any,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
        show_difference: bool = False,
    ) -> bool:
        if object1 is object2:
            return True
        if not isinstance(object2, ndarray):
            if show_difference:
                logger.info(f"object2 is not a numpy.ndarray: {type(object2)}")
            return False
        if self._check_dtype and object1.dtype != object2.dtype:
            if show_difference:
                logger.info(
                    f"numpy.ndarray data types are different: {object1.shape} vs {object2.shape}"
                )
            return False
        if object1.shape != object2.shape:
            if show_difference:
                logger.info(
                    f"numpy.ndarray shapes are different: {object1.shape} vs {object2.shape}"
                )
            return False
        object_equal = allclose(object1, object2, rtol=rtol, atol=atol, equal_nan=equal_nan)
        if show_difference and not object_equal:
            logger.info(f"numpy.ndarrays are different\nobject1=\n{object1}\nobject2=\n{object2}")
        return object_equal

    def clone(self) -> NDArrayAllCloseOperator:
        return self.__class__(check_dtype=self._check_dtype)


class NDArrayEqualityOperator(BaseEqualityOperator[ndarray]):
    r"""Implements an equality operator for ``numpy.ndarray``.

    Args:
        check_dtype (bool, optional): If ``True``, the data type of
            the arrays are checked, otherwise the data types are
            ignored. Default: ``True``
    """

    def __init__(self, check_dtype: bool = True) -> None:
        check_numpy()
        self._check_dtype = bool(check_dtype)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._check_dtype == other._check_dtype

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(check_dtype={self._check_dtype})"

    def clone(self) -> NDArrayEqualityOperator:
        return self.__class__(check_dtype=self._check_dtype)

    def equal(
        self,
        tester: BaseEqualityTester,
        object1: ndarray,
        object2: Any,
        show_difference: bool = False,
    ) -> bool:
        if object1 is object2:
            return True
        if not isinstance(object2, ndarray):
            if show_difference:
                logger.info(f"object2 is not a numpy.ndarray: {type(object2)}")
            return False
        if self._check_dtype and object1.dtype != object2.dtype:
            if show_difference:
                logger.info(
                    f"numpy.ndarray data types are different: {object1.shape} vs {object2.shape}"
                )
            return False
        if object1.shape != object2.shape:
            if show_difference:
                logger.info(
                    f"numpy.ndarray shapes are different: {object1.shape} vs {object2.shape}"
                )
            return False
        object_equal = array_equal(object1, object2)
        if show_difference and not object_equal:
            logger.info(f"numpy.ndarrays are different\nobject1=\n{object1}\nobject2=\n{object2}")
        return object_equal


def get_mapping_allclose() -> dict[type[object], BaseAllCloseOperator]:
    r"""Gets a default mapping between the types and the allclose
    operators.

    This function returns an empty dictionary if numpy is not
    installed.

    Returns:
        dict: The mapping between the types and the allclose
            operators.
    """
    if not is_numpy_available():
        return {}
    return {ndarray: NDArrayAllCloseOperator()}


def get_mapping_equality() -> dict[type[object], BaseEqualityOperator]:
    r"""Gets a default mapping between the types and the equality
    operators.

    This function returns an empty dictionary if numpy is not
    installed.

    Returns:
        dict: The mapping between the types and the equality
            operators.
    """
    if not is_numpy_available():
        return {}
    return {ndarray: NDArrayEqualityOperator()}
