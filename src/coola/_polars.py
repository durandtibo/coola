from __future__ import annotations

import logging
from typing import Any

from coola.equality import BaseEqualityOperator, BaseEqualityTester, EqualityTester
from coola.utils.imports import check_polars, is_polars_available

if is_polars_available():
    from polars import DataFrame, Series
else:
    DataFrame, Series = None, None  # pragma: no cover

logger = logging.getLogger(__name__)


class SeriesEqualityOperator(BaseEqualityOperator[Series]):
    r"""Implements an equality operator for ``polars.Series``."""

    def __init__(self) -> None:
        check_polars()

    def equal(
        self,
        tester: BaseEqualityTester,
        object1: Series,
        object2: Any,
        show_difference: bool = False,
    ) -> bool:
        if object1 is object2:
            return True
        if not isinstance(object2, Series):
            if show_difference:
                logger.info(f"object2 is not a polars.Series: {type(object2)}")
            return False
        try:
            from polars.testing import assert_series_equal

            assert_series_equal(object1, object2, check_exact=True, nans_compare_equal=False)
            object_equal = not object1.is_null().any()
        except AssertionError:
            object_equal = False
        if show_difference and not object_equal:
            logger.info(f"polars.Series are different\nobject1=\n{object1}\nobject2=\n{object2}")
        return object_equal


if is_polars_available():  # pragma: no cover
    if not EqualityTester.has_equality_operator(Series):
        EqualityTester.add_equality_operator(Series, SeriesEqualityOperator())
