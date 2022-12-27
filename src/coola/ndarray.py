__all__ = ["NDArrayEqualityOperator"]

import logging
from typing import Any

from coola.equal import BaseEqualityOperator, BaseEqualityTester, EqualityTester
from coola.import_utils import is_numpy_available

if is_numpy_available():
    from numpy import array_equal, ndarray
else:
    ndarray, array_equal = None, None  # pragma: no cover

logger = logging.getLogger(__name__)


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
    EqualityTester.add_equality_operator(ndarray, NDArrayEqualityOperator())
