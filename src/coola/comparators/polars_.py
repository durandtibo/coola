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
from coola.utils.imports import check_polars, is_polars_available

if TYPE_CHECKING:
    from coola.testers import BaseAllCloseTester, BaseEqualityTester

if is_polars_available():
    import polars
    from polars import DataFrame, Series
    from polars.testing import assert_frame_equal, assert_series_equal
else:
    DataFrame, Series = None, None  # pragma: no cover

logger = logging.getLogger(__name__)


class DataFrameAllCloseOperator(BaseAllCloseOperator[DataFrame]):
    r"""Implements an equality operator for ``polars.DataFrame``.

    Example usage:

    ```pycon
    >>> import polars
    >>> from coola.testers import AllCloseTester
    >>> from coola.comparators.polars_ import DataFrameAllCloseOperator
    >>> tester = AllCloseTester()
    >>> op = DataFrameAllCloseOperator()
    >>> op.allclose(
    ...     tester,
    ...     polars.DataFrame(
    ...         {
    ...             "col1": [1, 2, 3, 4, 5],
    ...             "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
    ...             "col3": ["a", "b", "c", "d", "e"],
    ...         }
    ...     ),
    ...     polars.DataFrame(
    ...         {
    ...             "col1": [1, 2, 3, 4, 5],
    ...             "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
    ...             "col3": ["a", "b", "c", "d", "e"],
    ...         }
    ...     ),
    ... )
    True

    ```
    """

    def __init__(self) -> None:
        check_polars()

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
                logger.info(f"object2 is not a polars.DataFrame: {type(object2)}")
            return False
        try:
            assert_frame_equal(
                object1,
                object2,
                rtol=rtol,
                atol=atol,
                check_exact=False,
            )
            object_equal = True
        except AssertionError:
            object_equal = False
        if not equal_nan and object_equal and has_nan(object1):
            object_equal = False
        if show_difference and not object_equal:
            logger.info(
                f"polars.DataFrames are different\nobject1=\n{object1}\nobject2=\n{object2}"
            )
        return object_equal


class DataFrameEqualityOperator(BaseEqualityOperator[DataFrame]):
    r"""Implements an equality operator for ``polars.DataFrame``.

    Example usage:

    ```pycon
    >>> import polars
    >>> from coola.comparators.polars_ import DataFrameEqualityOperator
    >>> from coola.testers import EqualityTester
    >>> tester = EqualityTester()
    >>> op = DataFrameEqualityOperator()
    >>> op.equal(
    ...     tester,
    ...     polars.DataFrame(
    ...         {
    ...             "col1": [1, 2, 3, 4, 5],
    ...             "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
    ...             "col3": ["a", "b", "c", "d", "e"],
    ...         }
    ...     ),
    ...     polars.DataFrame(
    ...         {
    ...             "col1": [1, 2, 3, 4, 5],
    ...             "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
    ...             "col3": ["a", "b", "c", "d", "e"],
    ...         }
    ...     ),
    ... )
    True

    ```
    """

    def __init__(self) -> None:
        check_polars()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def clone(self) -> DataFrameEqualityOperator:
        return self.__class__()

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
        object_equal = self._compare_dataframes(object1, object2)
        if show_difference and not object_equal:
            logger.info(
                f"polars.DataFrames are different\nobject1=\n{object1}\nobject2=\n{object2}"
            )
        return object_equal

    def _compare_dataframes(self, df1: DataFrame, df2: DataFrame) -> bool:
        r"""Indicates if the two DataFrames are equal or not.

        Args:
            df1 (``polars.DataFrame``): Specifies the first
                DataFrame to compare.
            df2 (``polars.DataFrame``): Specifies the second
                DataFrame to compare.

        Returns:
            bool: ``True``if the two series are equal,
                otherwise ``False``.
        """
        if has_nan(df1):
            return False
        try:
            assert_frame_equal(df1, df2, check_exact=True)
            object_equal = True
        except AssertionError:
            object_equal = False
        return object_equal


class SeriesAllCloseOperator(BaseAllCloseOperator[Series]):
    r"""Implements an equality operator for ``polars.Series``.

    Example usage:

    ```pycon
    >>> import polars
    >>> from coola.testers import AllCloseTester
    >>> from coola.comparators.polars_ import SeriesAllCloseOperator
    >>> tester = AllCloseTester()
    >>> op = SeriesAllCloseOperator()
    >>> op.allclose(tester, polars.Series([1, 2, 3, 4, 5]), polars.Series([1, 2, 3, 4, 5]))
    True

    ```
    """

    def __init__(self) -> None:
        check_polars()

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
                logger.info(f"object2 is not a polars.Series: {type(object2)}")
            return False
        try:
            assert_series_equal(
                object1,
                object2,
                rtol=rtol,
                atol=atol,
                check_exact=False,
            )
            object_equal = True
        except AssertionError:
            object_equal = False
        if not equal_nan and object_equal and has_nan(object1):
            object_equal = False
        if show_difference and not object_equal:
            logger.info(f"polars.Series are different\nobject1=\n{object1}\nobject2=\n{object2}")
        return object_equal


class SeriesEqualityOperator(BaseEqualityOperator[Series]):
    r"""Implements an equality operator for ``polars.Series``.

    Example usage:

    ```pycon
    >>> import polars
    >>> from coola.comparators.polars_ import SeriesEqualityOperator
    >>> from coola.testers import EqualityTester
    >>> tester = EqualityTester()
    >>> op = SeriesEqualityOperator()
    >>> op.equal(tester, polars.Series([1, 2, 3, 4, 5]), polars.Series([1, 2, 3, 4, 5]))
    True

    ```
    """

    def __init__(self) -> None:
        check_polars()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def clone(self) -> SeriesEqualityOperator:
        return self.__class__()

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
        object_equal = self._compare_series(object1, object2)
        if show_difference and not object_equal:
            logger.info(f"polars.Series are different\nobject1=\n{object1}\nobject2=\n{object2}")
        return object_equal

    def _compare_series(self, series1: Series, series2: Series) -> bool:
        r"""Indicates if the two series are equal or not.

        Args:
            series1 (``polars.Series``): Specifies the first series
                to compare.
            series2 (``polars.Series``): Specifies the second series
                to compare.

        Returns:
            bool: ``True``if the two series are equal,
                otherwise ``False``.
        """
        if has_nan(series1):
            return False
        try:
            assert_series_equal(series1, series2, check_exact=True)
            object_equal = True
        except AssertionError:
            object_equal = False
        return object_equal


def get_mapping_allclose() -> dict[type[object], BaseAllCloseOperator]:
    r"""Gets a default mapping between the types and the allclose
    operators.

    This function returns an empty dictionary if polars is not
    installed.

    Returns:
        dict: The mapping between the types and the allclose
            operators.
    """
    if not is_polars_available():
        return {}
    return {DataFrame: DataFrameAllCloseOperator(), Series: SeriesAllCloseOperator()}


def get_mapping_equality() -> dict[type[object], BaseEqualityOperator]:
    r"""Gets a default mapping between the types and the equality
    operators.

    This function returns an empty dictionary if polars is not
    installed.

    Returns:
        dict: The mapping between the types and the equality
            operators.
    """
    if not is_polars_available():
        return {}
    return {DataFrame: DataFrameEqualityOperator(), Series: SeriesEqualityOperator()}


def has_nan(obj: DataFrame | Series) -> bool:
    r"""Indicates if a DataFrame or Series has NaN values.

    Args:
        obj (``polars.DataFrame`` or ``polars.Series``): Specifies the
            object to check.

    Returns:
        bool: ``True`` if the DataFrame or Series has NaN values,
            otherwise ``False``.
    """
    if isinstance(obj, Series):
        return obj.dtype in polars.FLOAT_DTYPES and obj.is_nan().any()
    for col in obj:
        if col.dtype in polars.FLOAT_DTYPES and col.is_nan().any():
            return True
    return False
