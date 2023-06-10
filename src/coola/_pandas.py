from __future__ import annotations

import logging
from typing import Any

from coola.equality import BaseEqualityOperator, BaseEqualityTester, EqualityTester
from coola.utils.imports import check_pandas, is_pandas_available

if is_pandas_available():
    from pandas import DataFrame
else:
    DataFrame = None  # pragma: no cover

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
        print(object_equal)
        # try:
        #     pandas.testing.assert_frame_equal(
        #         object1,
        #         object2,
        #         check_index_type=True,
        #         check_column_type=True,
        #         check_exact=True,
        #     )
        #     object_equal = True
        # except AssertionError:
        #     object_equal = False
        if show_difference and not object_equal:
            logger.info(
                f"pandas.DataFrames are different\nobject1=\n{object1}\nobject2=\n{object2}"
            )
        return object_equal


if is_pandas_available():  # pragma: no cover
    if not EqualityTester.has_equality_operator(DataFrame):
        EqualityTester.add_equality_operator(DataFrame, DataFrameEqualityOperator())
