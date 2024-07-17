r"""Implement an equality comparator for ``polars.DataFrame``s and
``polars.Series``s."""

from __future__ import annotations

__all__ = [
    "PolarsDataFrameEqualityComparator",
    "PolarsSeriesEqualityComparator",
    "get_type_comparator_mapping",
]

import logging
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

from coola.equality.comparators.base import BaseEqualityComparator
from coola.equality.handlers import (
    PolarsDataFrameEqualHandler,
    PolarsSeriesEqualHandler,
    SameObjectHandler,
    SameTypeHandler,
)
from coola.utils import check_polars, is_polars_available

if is_polars_available():
    import polars as pl
else:  # pragma: no cover
    pl = Mock()

if TYPE_CHECKING:
    from coola.equality import EqualityConfig

logger = logging.getLogger(__name__)


class PolarsDataFrameEqualityComparator(BaseEqualityComparator[pl.DataFrame]):
    r"""Implement an equality comparator for ``polars.DataFrame``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.comparators import PolarsDataFrameEqualityComparator
    >>> from coola.equality.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> comparator = PolarsDataFrameEqualityComparator()
    >>> comparator.equal(
    ...     pl.DataFrame({"col": [1, 2, 3]}),
    ...     pl.DataFrame({"col": [1, 2, 3]}),
    ...     config,
    ... )
    True
    >>> comparator.equal(
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

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> PolarsDataFrameEqualityComparator:
        return self.__class__()

    def equal(self, actual: pl.DataFrame, expected: Any, config: EqualityConfig) -> bool:
        return self._handler.handle(actual, expected, config=config)


class PolarsSeriesEqualityComparator(BaseEqualityComparator[pl.Series]):
    r"""Implement an equality comparator for ``polars.Series``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.comparators import PolarsSeriesEqualityComparator
    >>> from coola.equality.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> comparator = PolarsSeriesEqualityComparator()
    >>> comparator.equal(pl.Series([1, 2, 3]), pl.Series([1, 2, 3]), config)
    True
    >>> comparator.equal(pl.Series([1, 2, 3]), pl.Series([1, 2, 4]), config)
    False

    ```
    """

    def __init__(self) -> None:
        check_polars()
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(PolarsSeriesEqualHandler())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> PolarsSeriesEqualityComparator:
        return self.__class__()

    def equal(self, actual: pl.Series, expected: Any, config: EqualityConfig) -> bool:
        return self._handler.handle(actual, expected, config=config)


def get_type_comparator_mapping() -> dict[type, BaseEqualityComparator]:
    r"""Get a default mapping between the types and the equality
    comparators.

    This function returns an empty dictionary if ``polars`` is not
    installed.

    Returns:
        The mapping between the types and the equality comparators.

    Example usage:

    ```pycon

    >>> from coola.equality.comparators.polars_ import get_type_comparator_mapping
    >>> get_type_comparator_mapping()
    {<class 'polars...DataFrame'>: PolarsDataFrameEqualityComparator(),
     <class 'polars...Series'>: PolarsSeriesEqualityComparator()}

    ```
    """
    if not is_polars_available():
        return {}
    return {
        pl.DataFrame: PolarsDataFrameEqualityComparator(),
        pl.Series: PolarsSeriesEqualityComparator(),
    }
