from __future__ import annotations

__all__ = ["NDArrayAllCloseOperator", "NDArrayEqualityOperator"]

import logging
from typing import Any

from coola.allclose import AllCloseTester, BaseAllCloseOperator, BaseAllCloseTester
from coola.equality import BaseEqualityOperator, BaseEqualityTester, EqualityTester
from coola.utils import check_numpy, is_numpy_available

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


if is_numpy_available():  # pragma: no cover
    if not AllCloseTester.has_operator(ndarray):
        AllCloseTester.add_operator(ndarray, NDArrayAllCloseOperator())
    if not EqualityTester.has_operator(ndarray):
        EqualityTester.add_operator(ndarray, NDArrayEqualityOperator())
