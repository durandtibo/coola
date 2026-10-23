r"""Implement equality testers for Polars DataFrames, LazyFrames, and
Series.

This module provides equality testers for polars.DataFrame,
polars.LazyFrame, and polars.Series using Polars' built-in equality
testing methods.
"""

from __future__ import annotations

__all__ = [
    "PolarsDataFrameEqualityTester",
    "PolarsLazyFrameEqualityTester",
    "PolarsSeriesEqualityTester",
]

from typing import TYPE_CHECKING, Any

from coola.equality.handler import (
    PolarsDataFrameEqualHandler,
    PolarsLazyFrameEqualHandler,
    PolarsSeriesEqualHandler,
    SameObjectHandler,
    SameTypeHandler,
    create_chain,
)
from coola.equality.tester.chain import BaseChainEqualityTester
from coola.utils.imports import check_polars, is_polars_available

if TYPE_CHECKING or is_polars_available():
    import polars as pl
else:  # pragma: no cover
    from coola.utils.fallback.polars import polars as pl

if TYPE_CHECKING:
    from coola.equality.handler import BaseEqualityHandler
    from coola.equality.tester.base import BaseEqualityTester


class PolarsDataFrameEqualityTester(BaseChainEqualityTester[pl.DataFrame]):
    r"""Implement an equality tester for ``polars.DataFrame``.

    This tester uses Polars' DataFrame equality testing. The handler chain:
    1. SameObjectHandler: Check for object identity
    2. SameTypeHandler: Verify both are polars DataFrames
    3. PolarsDataFrameEqualHandler: Use Polars' internal equality testing

    The tester respects config.equal_nan for NaN comparisons and config.atol/rtol
    for floating-point tolerance.

    Example:
        Basic DataFrame comparison:

        ```pycon
        >>> import polars as pl
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import PolarsDataFrameEqualityTester
        >>> config = EqualityConfig()
        >>> tester = PolarsDataFrameEqualityTester()
        >>> tester.objects_are_equal(
        ...     pl.DataFrame({"col": [1, 2, 3]}),
        ...     pl.DataFrame({"col": [1, 2, 3]}),
        ...     config,
        ... )
        True
        >>> tester.objects_are_equal(
        ...     pl.DataFrame({"col": [1, 2, 3]}),
        ...     pl.DataFrame({"col": [1, 2, 4]}),
        ...     config,
        ... )
        False

        ```
    """

    def _check_dependencies(self) -> None:
        check_polars()

    def _create_handler(self) -> BaseEqualityHandler:
        return create_chain(SameObjectHandler(), SameTypeHandler(), PolarsDataFrameEqualHandler())


class PolarsLazyFrameEqualityTester(BaseChainEqualityTester[pl.LazyFrame]):
    r"""Implement an equality tester for ``polars.LazyFrame``.

    This tester uses Polars' LazyFrame equality testing. The handler chain:
    1. SameObjectHandler: Check for object identity
    2. SameTypeHandler: Verify both are polars LazyFrames
    3. PolarsLazyFrameEqualHandler: Use Polars' internal equality testing

    The tester respects config.equal_nan for NaN comparisons and config.atol/rtol
    for floating-point tolerance.

    Note:
        LazyFrames represent query plans and are collected (materialized) for
        comparison, which may have performance implications for large datasets.

    Example:
        Basic LazyFrame comparison:

        ```pycon
        >>> import polars as pl
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import PolarsLazyFrameEqualityTester
        >>> config = EqualityConfig()
        >>> tester = PolarsLazyFrameEqualityTester()
        >>> tester.objects_are_equal(
        ...     pl.LazyFrame({"col": [1, 2, 3]}),
        ...     pl.LazyFrame({"col": [1, 2, 3]}),
        ...     config,
        ... )
        True
        >>> tester.objects_are_equal(
        ...     pl.LazyFrame({"col": [1, 2, 3]}),
        ...     pl.LazyFrame({"col": [1, 2, 4]}),
        ...     config,
        ... )
        False

        ```
    """

    def _check_dependencies(self) -> None:
        check_polars()

    def _create_handler(self) -> BaseEqualityHandler:
        return create_chain(SameObjectHandler(), SameTypeHandler(), PolarsLazyFrameEqualHandler())


class PolarsSeriesEqualityTester(BaseChainEqualityTester[pl.Series]):
    r"""Implement an equality tester for ``polars.Series``.

    This tester uses Polars' Series equality testing. The handler chain:
    1. SameObjectHandler: Check for object identity
    2. SameTypeHandler: Verify both are polars Series
    3. PolarsSeriesEqualHandler: Use Polars' internal equality testing

    The tester respects config.equal_nan for NaN comparisons and config.atol/rtol
    for floating-point tolerance.

    Example:
        Basic Series comparison:

        ```pycon
        >>> import polars as pl
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import PolarsSeriesEqualityTester
        >>> config = EqualityConfig()
        >>> tester = PolarsSeriesEqualityTester()
        >>> tester.objects_are_equal(pl.Series([1, 2, 3]), pl.Series([1, 2, 3]), config)
        True
        >>> tester.objects_are_equal(pl.Series([1, 2, 3]), pl.Series([1, 2, 4]), config)
        False

        ```
    """

    def _check_dependencies(self) -> None:
        check_polars()

    def _create_handler(self) -> BaseEqualityHandler:
        return create_chain(SameObjectHandler(), SameTypeHandler(), PolarsSeriesEqualHandler())


def get_default_testers() -> dict[type, BaseEqualityTester[Any]]:
    r"""Get the default equality testers of the polars objects.

    Returns:
        The equality testers of the polars objects and their associated types.

    Notes:
        This function is called by the default registry the first time it
        needs the equality testers of the polars objects.
    """
    if not is_polars_available():
        return {}

    return {
        pl.DataFrame: PolarsDataFrameEqualityTester(),
        pl.LazyFrame: PolarsLazyFrameEqualityTester(),
        pl.Series: PolarsSeriesEqualityTester(),
    }
