r"""Implement an equality tester for ``pandas.DataFrame``s and
``pandas.Series``s."""

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
    from coola.equality.config import EqualityConfig2


class PandasDataFrameEqualityTester(BaseEqualityTester[pd.DataFrame]):
    r"""Implement an equality tester for ``pandas.DataFrame``.

    Example:
        ```pycon
        >>> import pandas as pd
        >>> from coola.equality.config import EqualityConfig2
        >>> from coola.equality.tester import PandasDataFrameEqualityTester
        >>> config = EqualityConfig2()
        >>> tester = PandasDataFrameEqualityTester()
        >>> tester.equal(
        ...     pd.DataFrame({"col": [1, 2, 3]}),
        ...     pd.DataFrame({"col": [1, 2, 3]}),
        ...     config,
        ... )
        True
        >>> tester.equal(
        ...     pd.DataFrame({"col": [1, 2, 3]}),
        ...     pd.DataFrame({"col": [1, 2, 4]}),
        ...     config,
        ... )
        False

        ```
    """

    def __init__(self) -> None:
        check_pandas()
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(PandasDataFrameEqualHandler())

    def equal(self, other: object) -> bool:
        return type(other) is type(self)

    def objects_are_equal(
        self,
        actual: pd.DataFrame,
        expected: object,
        config: EqualityConfig2,
    ) -> bool:
        return self._handler.handle(actual, expected, config=config)


class PandasSeriesEqualityTester(BaseEqualityTester[pd.Series]):
    r"""Implement an equality tester for ``pandas.Series``.

    Example:
        ```pycon
        >>> import pandas as pd
        >>> from coola.equality.config import EqualityConfig2
        >>> from coola.equality.tester import PandasSeriesEqualityTester
        >>> config = EqualityConfig2()
        >>> tester = PandasSeriesEqualityTester()
        >>> tester.equal(pd.Series([1, 2, 3]), pd.Series([1, 2, 3]), config)
        True
        >>> tester.equal(pd.Series([1, 2, 3]), pd.Series([1, 2, 4]), config)
        False

        ```
    """

    def __init__(self) -> None:
        check_pandas()
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(PandasSeriesEqualHandler())

    def equal(self, other: object) -> bool:
        return type(other) is type(self)

    def objects_are_equal(
        self,
        actual: pd.Series,
        expected: object,
        config: EqualityConfig2,
    ) -> bool:
        return self._handler.handle(actual, expected, config=config)
