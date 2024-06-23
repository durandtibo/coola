r"""Implement some handlers for ``polars.DataFrame``s and
``polars.Series``s."""

from __future__ import annotations

__all__ = ["PolarsDataFrameEqualHandler", "PolarsSeriesEqualHandler"]

import logging
from typing import TYPE_CHECKING
from unittest.mock import Mock

from coola.equality.handlers.base import BaseEqualityHandler
from coola.utils import is_polars_available

if is_polars_available():
    import polars
    from polars.testing import assert_frame_equal, assert_series_equal
else:  # pragma: no cover
    polars = Mock()

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig

logger = logging.getLogger(__name__)


class PolarsDataFrameEqualHandler(BaseEqualityHandler):
    r"""Check if the two ``polars.DataFrame`` are equal.

    This handler returns ``True`` if the two ``polars.DataFrame``s
    equal, otherwise ``False``. This handler is designed to be used
    at the end of the chain of responsibility. This handler does
    not call the next handler.

    Example usage:

    ```pycon

    >>> import polars
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.handlers import PolarsDataFrameEqualHandler
    >>> from coola.equality.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> handler = PolarsDataFrameEqualHandler()
    >>> handler.handle(
    ...     polars.DataFrame({"col": [1, 2, 3]}),
    ...     polars.DataFrame({"col": [1, 2, 3]}),
    ...     config,
    ... )
    True
    >>> handler.handle(
    ...     polars.DataFrame({"col": [1, 2, 3]}),
    ...     polars.DataFrame({"col": [1, 2, 4]}),
    ...     config,
    ... )
    False

    ```
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def handle(
        self,
        actual: polars.DataFrame,
        expected: polars.DataFrame,
        config: EqualityConfig,
    ) -> bool:
        object_equal = frame_equal(actual, expected, config)
        if config.show_difference and not object_equal:
            logger.info(
                f"polars.DataFrames have different elements:\n"
                f"actual:\n{actual}\nexpected:\n{expected}"
            )
        return object_equal

    def set_next_handler(self, handler: BaseEqualityHandler) -> None:
        pass  # Do nothing because the next handler is never called.


class PolarsSeriesEqualHandler(BaseEqualityHandler):
    r"""Check if the two ``polars.Series`` are equal.

    This handler returns ``True`` if the two arrays ``polars.Series``
    equal, otherwise ``False``. This handler is designed to be used
    at the end of the chain of responsibility. This handler does
    not call the next handler.

    Example usage:

    ```pycon

    >>> import polars
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.handlers import PolarsSeriesEqualHandler
    >>> from coola.equality.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> handler = PolarsSeriesEqualHandler()
    >>> handler.handle(polars.Series([1, 2, 3]), polars.Series([1, 2, 3]), config)
    True
    >>> handler.handle(polars.Series([1, 2, 3]), polars.Series([1, 2, 4]), config)
    False

    ```
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def handle(
        self,
        actual: polars.Series,
        expected: polars.Series,
        config: EqualityConfig,
    ) -> bool:
        object_equal = series_equal(actual, expected, config)
        if config.show_difference and not object_equal:
            logger.info(
                f"polars.Series have different elements:\n"
                f"actual:\n{actual}\nexpected:\n{expected}"
            )
        return object_equal

    def set_next_handler(self, handler: BaseEqualityHandler) -> None:
        pass  # Do nothing because the next handler is never called.


def has_nan(df_or_series: polars.DataFrame | polars.Series) -> bool:
    r"""Indicate if a DataFrame or Series has NaN values.

    Args:
        df_or_series: The DataFrame or series to check.

    Returns:
        ``True`` if the DataFrame or Series has NaN values,
            otherwise ``False``.
    """
    if isinstance(df_or_series, polars.Series):
        return df_or_series.dtype in polars.FLOAT_DTYPES and df_or_series.is_nan().any()
    return any(col.dtype in polars.FLOAT_DTYPES and col.is_nan().any() for col in df_or_series)


def frame_equal(df1: polars.DataFrame, df2: polars.DataFrame, config: EqualityConfig) -> bool:
    r"""Indicate if the two DataFrames are equal or not.

    Args:
        df1: The first DataFrame to compare.
        df2: The second DataFrame to compare.
        config: The equality configuration.

    Returns:
        ``True``if the two DataFrame are equal, otherwise ``False``.
    """
    if not config.equal_nan and has_nan(df1):
        return False
    try:
        assert_frame_equal(
            df1,
            df2,
            check_exact=config.atol == 0 and config.rtol == 0,
            atol=config.atol,
            rtol=config.rtol,
        )
    except AssertionError:
        return False
    return True


def series_equal(series1: polars.Series, series2: polars.Series, config: EqualityConfig) -> bool:
    r"""Indicate if the two series are equal or not.

    Args:
        series1: The first series to compare.
        series2: The second series to compare.
        config: The equality configuration.

    Returns:
        ``True``if the two series are equal, otherwise ``False``.
    """
    if not config.equal_nan and has_nan(series1):
        return False
    try:
        assert_series_equal(
            series1,
            series2,
            check_exact=config.atol == 0 and config.rtol == 0,
            atol=config.atol,
            rtol=config.rtol,
        )
    except AssertionError:
        return False
    return True
