r"""Implement an equality tester for ``polars.DataFrame``s and
``polars.Series``s."""

from __future__ import annotations

__all__ = [
    "PolarsDataFrameEqualityTester",
    "PolarsLazyFrameEqualityTester",
    "PolarsSeriesEqualityTester",
]

from typing import TYPE_CHECKING

from coola.equality.handler import (
    PolarsDataFrameEqualHandler,
    PolarsLazyFrameEqualHandler,
    PolarsSeriesEqualHandler,
    SameObjectHandler,
    SameTypeHandler,
)
from coola.equality.tester.base import BaseEqualityTester
from coola.utils.imports import check_polars, is_polars_available

if TYPE_CHECKING or is_polars_available():
    import polars as pl
else:  # pragma: no cover
    from coola.utils.fallback.polars import polars as pl

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig2


class PolarsDataFrameEqualityTester(BaseEqualityTester[pl.DataFrame]):
    r"""Implement an equality tester for ``polars.DataFrame``.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from coola.equality.config import EqualityConfig2
        >>> from coola.equality.tester import PolarsDataFrameEqualityTester
        >>> config = EqualityConfig2()
        >>> tester = PolarsDataFrameEqualityTester()
        >>> tester.equal(
        ...     pl.DataFrame({"col": [1, 2, 3]}),
        ...     pl.DataFrame({"col": [1, 2, 3]}),
        ...     config,
        ... )
        True
        >>> tester.equal(
        ...     pl.DataFrame({"col": [1, 2, 3]}),
        ...     pl.DataFrame({"col": [1, 2, 4]}),
        ...     config,
        ... )
        False

        ```
    """

    def __init__(self) -> None:
        check_polars()
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(PolarsDataFrameEqualHandler())

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def equal(self, other: object) -> bool:
        return type(other) is type(self)

    def objects_are_equal(
        self,
        actual: pl.DataFrame,
        expected: object,
        config: EqualityConfig2,
    ) -> bool:
        return self._handler.handle(actual, expected, config=config)


class PolarsLazyFrameEqualityTester(BaseEqualityTester[pl.LazyFrame]):
    r"""Implement an equality tester for ``polars.LazyFrame``.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from coola.equality.config import EqualityConfig2
        >>> from coola.equality.tester import PolarsLazyFrameEqualityTester
        >>> config = EqualityConfig2()
        >>> tester = PolarsLazyFrameEqualityTester()
        >>> tester.equal(
        ...     pl.LazyFrame({"col": [1, 2, 3]}),
        ...     pl.LazyFrame({"col": [1, 2, 3]}),
        ...     config,
        ... )
        True
        >>> tester.equal(
        ...     pl.LazyFrame({"col": [1, 2, 3]}),
        ...     pl.LazyFrame({"col": [1, 2, 4]}),
        ...     config,
        ... )
        False

        ```
    """

    def __init__(self) -> None:
        check_polars()
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(PolarsLazyFrameEqualHandler())

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def equal(self, other: object) -> bool:
        return type(other) is type(self)

    def objects_are_equal(
        self,
        actual: pl.LazyFrame,
        expected: object,
        config: EqualityConfig2,
    ) -> bool:
        return self._handler.handle(actual, expected, config=config)


class PolarsSeriesEqualityTester(BaseEqualityTester[pl.Series]):
    r"""Implement an equality tester for ``polars.Series``.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from coola.equality.config import EqualityConfig2
        >>> from coola.equality.tester import PolarsSeriesEqualityTester
        >>> config = EqualityConfig2()
        >>> tester = PolarsSeriesEqualityTester()
        >>> tester.equal(pl.Series([1, 2, 3]), pl.Series([1, 2, 3]), config)
        True
        >>> tester.equal(pl.Series([1, 2, 3]), pl.Series([1, 2, 4]), config)
        False

        ```
    """

    def __init__(self) -> None:
        check_polars()
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(PolarsSeriesEqualHandler())

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def equal(self, other: object) -> bool:
        return type(other) is type(self)

    def objects_are_equal(
        self,
        actual: pl.Series,
        expected: object,
        config: EqualityConfig2,
    ) -> bool:
        return self._handler.handle(actual, expected, config=config)
