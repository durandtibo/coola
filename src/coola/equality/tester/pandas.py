r"""Implement equality testers for pandas DataFrames and Series.

This module provides equality testers for pandas.DataFrame and
pandas.Series using pandas' built-in equality testing methods.
"""

from __future__ import annotations

__all__ = ["PandasDataFrameEqualityTester", "PandasSeriesEqualityTester"]

from typing import TYPE_CHECKING

from coola.equality.handler import (
    PandasDataFrameEqualHandler,
    PandasSeriesEqualHandler,
    SameObjectHandler,
    SameTypeHandler,
)
from coola.equality.tester.base import BaseEqualityTester
from coola.utils.imports import check_pandas, is_pandas_available

if TYPE_CHECKING or is_pandas_available():
    import pandas as pd
else:  # pragma: no cover
    from coola.utils.fallback.pandas import pandas as pd

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig


class PandasDataFrameEqualityTester(BaseEqualityTester[pd.DataFrame]):
    r"""Implement an equality tester for ``pandas.DataFrame``.

    This tester uses pandas' DataFrame equality testing which compares shape,
    column names, data types, index, and values. The handler chain:
    1. SameObjectHandler: Check for object identity
    2. SameTypeHandler: Verify both are pandas DataFrames
    3. PandasDataFrameEqualHandler: Use pandas' assert_frame_equal internally

    Note:
        The tester uses pandas' internal comparison logic which handles NaN values
        and performs comprehensive DataFrame equality checking.

    Example:
        Basic DataFrame comparison:

        ```pycon
        >>> import pandas as pd
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import PandasDataFrameEqualityTester
        >>> config = EqualityConfig()
        >>> tester = PandasDataFrameEqualityTester()
        >>> tester.objects_are_equal(
        ...     pd.DataFrame({"col": [1, 2, 3]}),
        ...     pd.DataFrame({"col": [1, 2, 3]}),
        ...     config,
        ... )
        True
        >>> tester.objects_are_equal(
        ...     pd.DataFrame({"col": [1, 2, 3]}),
        ...     pd.DataFrame({"col": [1, 2, 4]}),
        ...     config,
        ... )
        False

        ```

        Different column names are not equal:

        ```pycon
        >>> import pandas as pd
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import PandasDataFrameEqualityTester
        >>> config = EqualityConfig()
        >>> tester = PandasDataFrameEqualityTester()
        >>> tester.objects_are_equal(
        ...     pd.DataFrame({"col1": [1, 2, 3]}),
        ...     pd.DataFrame({"col2": [1, 2, 3]}),
        ...     config,
        ... )
        False

        ```
    """

    def __init__(self) -> None:
        check_pandas()
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(PandasDataFrameEqualHandler())

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def equal(self, other: object) -> bool:
        return type(other) is type(self)

    def objects_are_equal(
        self,
        actual: pd.DataFrame,
        expected: object,
        config: EqualityConfig,
    ) -> bool:
        return self._handler.handle(actual, expected, config=config)


class PandasSeriesEqualityTester(BaseEqualityTester[pd.Series]):
    r"""Implement an equality tester for ``pandas.Series``.

    This tester uses pandas' Series equality testing which compares length,
    data type, index, and values. The handler chain:
    1. SameObjectHandler: Check for object identity
    2. SameTypeHandler: Verify both are pandas Series
    3. PandasSeriesEqualHandler: Use pandas' assert_series_equal internally

    Note:
        The tester uses pandas' internal comparison logic which handles NaN values
        and performs comprehensive Series equality checking.

    Example:
        Basic Series comparison:

        ```pycon
        >>> import pandas as pd
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import PandasSeriesEqualityTester
        >>> config = EqualityConfig()
        >>> tester = PandasSeriesEqualityTester()
        >>> tester.objects_are_equal(pd.Series([1, 2, 3]), pd.Series([1, 2, 3]), config)
        True
        >>> tester.objects_are_equal(pd.Series([1, 2, 3]), pd.Series([1, 2, 4]), config)
        False

        ```

        Different index values are not equal:

        ```pycon
        >>> import pandas as pd
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import PandasSeriesEqualityTester
        >>> config = EqualityConfig()
        >>> tester = PandasSeriesEqualityTester()
        >>> tester.objects_are_equal(
        ...     pd.Series([1, 2, 3], index=["a", "b", "c"]),
        ...     pd.Series([1, 2, 3], index=["x", "y", "z"]),
        ...     config,
        ... )
        False

        ```
    """

    def __init__(self) -> None:
        check_pandas()
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(PandasSeriesEqualHandler())

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def equal(self, other: object) -> bool:
        return type(other) is type(self)

    def objects_are_equal(
        self,
        actual: pd.Series,
        expected: object,
        config: EqualityConfig,
    ) -> bool:
        return self._handler.handle(actual, expected, config=config)
