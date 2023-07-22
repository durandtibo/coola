from __future__ import annotations

__all__ = [
    "DataFrameAllCloseOperator",
    "DataFrameEqualityOperator",
    "SeriesAllCloseOperator",
    "SeriesEqualityOperator",
    "get_mapping_allclose",
    "get_mapping_equality",
]

import logging
from typing import TYPE_CHECKING, Any

from coola.comparators.base import BaseAllCloseOperator, BaseEqualityOperator
from coola.utils.imports import check_pandas, is_pandas_available

if TYPE_CHECKING:
    from coola.testers import BaseAllCloseTester, BaseEqualityTester

if is_pandas_available():
    from pandas import DataFrame, Series
    from pandas.testing import assert_frame_equal, assert_series_equal
else:
    DataFrame, Series = None, None  # pragma: no cover

logger = logging.getLogger(__name__)


class DataFrameAllCloseOperator(BaseAllCloseOperator[DataFrame]):
    r"""Implements an equality operator for ``pandas.DataFrame``."""

    def __init__(self) -> None:
        check_pandas()

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> DataFrameAllCloseOperator:
        return self.__class__()

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
                logger.info(f"object2 is not a pandas.DataFrame: {type(object2)}")
            return False
        try:
            assert_frame_equal(object1, object2, rtol=rtol, atol=atol)
            object_equal = True
        except AssertionError:
            object_equal = False
        if not equal_nan and object_equal:
            object_equal = not object1.isnull().any().any()
        if show_difference and not object_equal:
            logger.info(
                f"pandas.DataFrames are different\nobject1=\n{object1}\nobject2=\n{object2}"
            )
        return object_equal


class DataFrameEqualityOperator(BaseEqualityOperator[DataFrame]):
    r"""Implements an equality operator for ``pandas.DataFrame``.

    Args:
        nulls_compare_equal (bool, optional): If ``True``, null values
            (e.g. NaN or NaT) compare as true. Default: ``False``
    """

    def __init__(self, nulls_compare_equal: bool = False) -> None:
        check_pandas()
        self._nulls_compare_equal = bool(nulls_compare_equal)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._nulls_compare_equal == other._nulls_compare_equal

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(nulls_compare_equal={self._nulls_compare_equal})"

    def clone(self) -> DataFrameEqualityOperator:
        return self.__class__(nulls_compare_equal=self._nulls_compare_equal)

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
        object_equal = self._compare_dataframes(object1, object2)
        if show_difference and not object_equal:
            logger.info(
                f"pandas.DataFrames are different\nobject1=\n{object1}\nobject2=\n{object2}"
            )
        return object_equal

    def _compare_dataframes(self, df1: DataFrame, df2: DataFrame) -> bool:
        r"""Indicates if the two DataFrames are equal or not.

        Args:
            df1 (``pandas.DataFrame``): Specifies the first
                DataFrame to compare.
            df2 (``pandas.DataFrame``): Specifies the second
                DataFrame to compare.

        Returns:
            bool: ``True``if the two series are equal,
                otherwise ``False``.
        """
        try:
            assert_frame_equal(df1, df2, check_exact=True)
            object_equal = True
        except AssertionError:
            object_equal = False
        if object_equal and not self._nulls_compare_equal:
            object_equal = not df1.isnull().any().any()
        return object_equal


class SeriesAllCloseOperator(BaseAllCloseOperator[Series]):
    r"""Implements an equality operator for ``pandas.Series``."""

    def __init__(self) -> None:
        check_pandas()

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> SeriesAllCloseOperator:
        return self.__class__()

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
        try:
            assert_series_equal(object1, object2, rtol=rtol, atol=atol)
            object_equal = True
        except AssertionError:
            object_equal = False
        if not equal_nan and object_equal and object1.isnull().any():
            object_equal = False
        if show_difference and not object_equal:
            logger.info(f"pandas.Series are different\nobject1=\n{object1}\nobject2=\n{object2}")
        return object_equal


class SeriesEqualityOperator(BaseEqualityOperator[Series]):
    r"""Implements an equality operator for ``pandas.Series``.

    Args:
        nulls_compare_equal (bool, optional): If ``True``, null values
            (e.g. NaN or NaT) compare as true. Default: ``False``
    """

    def __init__(self, nulls_compare_equal: bool = False) -> None:
        check_pandas()
        self._nulls_compare_equal = bool(nulls_compare_equal)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._nulls_compare_equal == other._nulls_compare_equal

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(nulls_compare_equal={self._nulls_compare_equal})"

    def clone(self) -> SeriesEqualityOperator:
        return self.__class__(nulls_compare_equal=self._nulls_compare_equal)

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
        object_equal = self._compare_series(object1, object2)
        if show_difference and not object_equal:
            logger.info(f"pandas.Series are different\nobject1=\n{object1}\nobject2=\n{object2}")
        return object_equal

    def _compare_series(self, series1: Series, series2: Series) -> bool:
        r"""Indicates if the two series are equal or not.

        Args:
            series1 (``pandas.Series``): Specifies the first series
                to compare.
            series2 (``pandas.Series``): Specifies the second series
                to compare.

        Returns:
            bool: ``True``if the two series are equal,
                otherwise ``False``.
        """
        try:
            assert_series_equal(series1, series2, check_exact=True)
            object_equal = True
        except AssertionError:
            object_equal = False
        if object_equal and not self._nulls_compare_equal:
            object_equal = not series1.isnull().any()
        return object_equal


def get_mapping_allclose() -> dict[type[object], BaseAllCloseOperator]:
    r"""Gets a default mapping between the types and the allclose
    operators.

    This function returns an empty dictionary if pandas is not
    installed.

    Returns:
        dict: The mapping between the types and the allclose
            operators.
    """
    if not is_pandas_available():
        return {}
    return {DataFrame: DataFrameAllCloseOperator(), Series: SeriesAllCloseOperator()}


def get_mapping_equality() -> dict[type[object], BaseEqualityOperator]:
    r"""Gets a default mapping between the types and the equality
    operators.

    This function returns an empty dictionary if pandas is not
    installed.

    Returns:
        dict: The mapping between the types and the equality
            operators.
    """
    if not is_pandas_available():
        return {}
    return {DataFrame: DataFrameEqualityOperator(), Series: SeriesEqualityOperator()}
