r"""Implement an equality comparator for ``pandas.DataFrame``s and
``pandas.Series``s."""

from __future__ import annotations

__all__ = [
    "PandasDataFrameEqualityComparator",
    "PandasSeriesEqualityComparator",
    "get_type_comparator_mapping",
]

import logging
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

from coola.equality.comparators.base import BaseEqualityComparator
from coola.equality.handlers import (
    PandasDataFrameEqualHandler,
    PandasSeriesEqualHandler,
    SameObjectHandler,
    SameTypeHandler,
)
from coola.utils import check_pandas, is_pandas_available

if is_pandas_available():
    import pandas as pd
else:  # pragma: no cover
    pd = Mock()

if TYPE_CHECKING:
    from coola.equality import EqualityConfig

logger = logging.getLogger(__name__)


class PandasDataFrameEqualityComparator(BaseEqualityComparator[pd.DataFrame]):
    r"""Implement an equality comparator for ``pandas.DataFrame``.

    Example usage:

    ```pycon

    >>> import pandas as pd
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.comparators import PandasDataFrameEqualityComparator
    >>> from coola.equality.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> comparator = PandasDataFrameEqualityComparator()
    >>> comparator.equal(
    ...     pd.DataFrame({"col": [1, 2, 3]}),
    ...     pd.DataFrame({"col": [1, 2, 3]}),
    ...     config,
    ... )
    True
    >>> comparator.equal(
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

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> PandasDataFrameEqualityComparator:
        return self.__class__()

    def equal(self, actual: pd.DataFrame, expected: Any, config: EqualityConfig) -> bool:
        return self._handler.handle(actual, expected, config=config)


class PandasSeriesEqualityComparator(BaseEqualityComparator[pd.Series]):
    r"""Implement an equality comparator for ``pandas.Series``.

    Example usage:

    ```pycon

    >>> import pandas as pd
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.comparators import PandasSeriesEqualityComparator
    >>> from coola.equality.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> comparator = PandasSeriesEqualityComparator()
    >>> comparator.equal(pd.Series([1, 2, 3]), pd.Series([1, 2, 3]), config)
    True
    >>> comparator.equal(pd.Series([1, 2, 3]), pd.Series([1, 2, 4]), config)
    False

    ```
    """

    def __init__(self) -> None:
        check_pandas()
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(PandasSeriesEqualHandler())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> PandasSeriesEqualityComparator:
        return self.__class__()

    def equal(self, actual: pd.Series, expected: Any, config: EqualityConfig) -> bool:
        return self._handler.handle(actual, expected, config=config)


def get_type_comparator_mapping() -> dict[type, BaseEqualityComparator]:
    r"""Get a default mapping between the types and the equality
    comparators.

    This function returns an empty dictionary if ``pandas`` is not
    installed.

    Returns:
        The mapping between the types and the equality comparators.

    Example usage:

    ```pycon

    >>> from coola.equality.comparators.pandas_ import get_type_comparator_mapping
    >>> get_type_comparator_mapping()
    {<class 'pandas...DataFrame'>: PandasDataFrameEqualityComparator(),
     <class 'pandas...Series'>: PandasSeriesEqualityComparator()}

    ```
    """
    if not is_pandas_available():
        return {}
    return {
        pd.DataFrame: PandasDataFrameEqualityComparator(),
        pd.Series: PandasSeriesEqualityComparator(),
    }
