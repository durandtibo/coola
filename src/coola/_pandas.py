from __future__ import annotations

import logging
from typing import Any

from coola.allclose import AllCloseTester, BaseAllCloseOperator, BaseAllCloseTester
from coola.equality import BaseEqualityOperator, BaseEqualityTester, EqualityTester
from coola.utils.imports import check_pandas, is_pandas_available

if is_pandas_available():
    from pandas import DataFrame, Series
else:
    DataFrame, Series = None, None  # pragma: no cover

logger = logging.getLogger(__name__)


class DataFrameEqualityOperator(BaseEqualityOperator[DataFrame]):
    r"""Implements an equality operator for ``pandas.DataFrame``."""

    def __init__(self) -> None:
        check_pandas()

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
                logger.info(f"object2 is not a pandas.DataFrame: {type(object2)}")
            return False
        object_equal = object1.eq(object2).all().all() and all(object1.dtypes == object2.dtypes)
        if show_difference and not object_equal:
            logger.info(
                f"pandas.DataFrames are different\nobject1=\n{object1}\nobject2=\n{object2}"
            )
        return object_equal


class SeriesAllCloseOperator(BaseAllCloseOperator[Series]):
    r"""Implements an equality operator for ``pandas.Series``."""

    def __init__(self) -> None:
        check_pandas()

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
                logger.info(f"object2 is not a pandas.Series: {type(object2)}")
            return False
        object_equal = tester.allclose(
            object1.to_numpy(),
            object2.to_numpy(),
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            show_difference=show_difference,
        )
        if show_difference and not object_equal:
            logger.info(f"pandas.Series are different\nobject1=\n{object1}\nobject2=\n{object2}")
        return object_equal


class SeriesEqualityOperator(BaseEqualityOperator[Series]):
    r"""Implements an equality operator for ``pandas.Series``."""

    def __init__(self) -> None:
        check_pandas()

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
                logger.info(f"object2 is not a pandas.Series: {type(object2)}")
            return False
        object_equal = object1.eq(object2).all() and object1.dtypes == object2.dtypes
        if show_difference and not object_equal:
            logger.info(f"pandas.Series are different\nobject1=\n{object1}\nobject2=\n{object2}")
        return object_equal


if is_pandas_available():  # pragma: no cover
    if not AllCloseTester.has_allclose_operator(Series):
        AllCloseTester.add_allclose_operator(Series, SeriesAllCloseOperator())

    if not EqualityTester.has_equality_operator(DataFrame):
        EqualityTester.add_equality_operator(DataFrame, DataFrameEqualityOperator())
    if not EqualityTester.has_equality_operator(Series):
        EqualityTester.add_equality_operator(Series, SeriesEqualityOperator())
