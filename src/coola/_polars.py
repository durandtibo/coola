from __future__ import annotations

import logging
from typing import Any

from coola.allclose import AllCloseTester, BaseAllCloseOperator, BaseAllCloseTester
from coola.equality import BaseEqualityOperator, BaseEqualityTester, EqualityTester
from coola.utils.imports import check_polars, is_polars_available

if is_polars_available():
    from polars import DataFrame, Series
    from polars.testing import assert_frame_equal, assert_series_equal
else:
    DataFrame, Series = None, None  # pragma: no cover

logger = logging.getLogger(__name__)


class DataFrameAllCloseOperator(BaseAllCloseOperator[DataFrame]):
    r"""Implements an equality operator for ``polars.DataFrame``."""

    def __init__(self) -> None:
        check_polars()

    def allclose(
        self,
        tester: BaseAllCloseTester,
        object1: DataFrame,
        object2: Any,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
        show_difference: bool = False,
    ) -> bool:
        if object1 is object2:
            return True
        if not isinstance(object2, DataFrame):
            if show_difference:
                logger.info(f"object2 is not a polars.DataFrame: {type(object2)}")
            return False
        try:
            assert_frame_equal(
                object1,
                object2,
                rtol=rtol,
                atol=atol,
                nans_compare_equal=equal_nan,
                check_exact=False,
            )
            object_equal = True
        except AssertionError:
            object_equal = False
        if not equal_nan and object_equal:
            object_equal = object1.null_count().sum(axis=1).to_list()[0] == 0
        if show_difference and not object_equal:
            logger.info(
                f"polars.DataFrames are different\nobject1=\n{object1}\nobject2=\n{object2}"
            )
        return object_equal


class DataFrameEqualityOperator(BaseEqualityOperator[DataFrame]):
    r"""Implements an equality operator for ``polars.DataFrame``."""

    def __init__(self) -> None:
        check_polars()

    def equal(
        self,
        tester: BaseEqualityTester,
        object1: DataFrame,
        object2: Any,
        show_difference: bool = False,
    ) -> bool:
        if object1 is object2:
            return True
        if not isinstance(object2, DataFrame):
            if show_difference:
                logger.info(f"object2 is not a polars.DataFrame: {type(object2)}")
            return False
        try:
            assert_frame_equal(object1, object2, check_exact=True, nans_compare_equal=False)
            object_equal = object1.null_count().sum(axis=1).to_list()[0] == 0
        except AssertionError:
            object_equal = False
        if show_difference and not object_equal:
            logger.info(
                f"polars.DataFrames are different\nobject1=\n{object1}\nobject2=\n{object2}"
            )
        return object_equal


class SeriesAllCloseOperator(BaseAllCloseOperator[Series]):
    r"""Implements an equality operator for ``polars.Series``."""

    def __init__(self) -> None:
        check_polars()

    def allclose(
        self,
        tester: BaseAllCloseTester,
        object1: Series,
        object2: Any,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
        show_difference: bool = False,
    ) -> bool:
        if object1 is object2:
            return True
        if not isinstance(object2, Series):
            if show_difference:
                logger.info(f"object2 is not a polars.Series: {type(object2)}")
            return False
        try:
            assert_series_equal(
                object1,
                object2,
                rtol=rtol,
                atol=atol,
                nans_compare_equal=equal_nan,
                check_exact=False,
            )
            object_equal = True
        except AssertionError:
            object_equal = False
        if not equal_nan and object_equal:
            object_equal = not object1.is_null().any()
        if show_difference and not object_equal:
            logger.info(f"polars.Series are different\nobject1=\n{object1}\nobject2=\n{object2}")
        return object_equal


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
            assert_series_equal(object1, object2, check_exact=True, nans_compare_equal=False)
            object_equal = not object1.is_null().any()
        except AssertionError:
            object_equal = False
        if show_difference and not object_equal:
            logger.info(f"polars.Series are different\nobject1=\n{object1}\nobject2=\n{object2}")
        return object_equal


if is_polars_available():  # pragma: no cover
    if not AllCloseTester.has_allclose_operator(DataFrame):
        AllCloseTester.add_allclose_operator(DataFrame, DataFrameAllCloseOperator())
    if not AllCloseTester.has_allclose_operator(Series):
        AllCloseTester.add_allclose_operator(Series, SeriesAllCloseOperator())

    if not EqualityTester.has_equality_operator(DataFrame):
        EqualityTester.add_equality_operator(DataFrame, DataFrameEqualityOperator())
    if not EqualityTester.has_equality_operator(Series):
        EqualityTester.add_equality_operator(Series, SeriesEqualityOperator())
