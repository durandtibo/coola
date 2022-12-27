__all__ = ["NDArrayAllCloseOperator", "NDArrayEqualityOperator"]

import logging
from typing import Any

from coola.allclose import AllCloseTester, BaseAllCloseOperator, BaseAllCloseTester
from coola.equality import BaseEqualityOperator, BaseEqualityTester, EqualityTester
from coola.import_utils import is_numpy_available

if is_numpy_available():
    from numpy import allclose, array_equal, ndarray
else:
    ndarray, array_equal, allclose = None, None, None  # pragma: no cover

logger = logging.getLogger(__name__)


class NDArrayAllCloseOperator(BaseAllCloseOperator[ndarray]):
    r"""Implements an allclose operator for ``numpy.ndarray``."""

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
        object_equal = allclose(object1, object2, rtol=rtol, atol=atol, equal_nan=equal_nan)
        if show_difference and not object_equal:
            logger.info(f"numpy.arrays are different\nobject1=\n{object1}\nobject2=\n{object2}")
        return object_equal


class NDArrayEqualityOperator(BaseEqualityOperator[ndarray]):
    r"""Implements an equality operator for ``numpy.ndarray``."""

    def equal(
        self,
        tester: BaseEqualityTester,
        object1: ndarray,
        object2: Any,
        show_difference: bool = False,
    ) -> bool:
        object_equal = array_equal(object1, object2)
        if show_difference and not object_equal:
            logger.info(f"numpy.arrays are different\nobject1=\n{object1}\nobject2=\n{object2}")
        return object_equal


if is_numpy_available() and not EqualityTester.has_equality_operator(ndarray):
    AllCloseTester.add_allclose_operator(ndarray, NDArrayAllCloseOperator())
    EqualityTester.add_equality_operator(ndarray, NDArrayEqualityOperator())
