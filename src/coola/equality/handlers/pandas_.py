r"""Implement some handlers for ``pandas.DataFrame``s and
``pandas.Series``s."""

from __future__ import annotations

__all__ = ["PandasSeriesEqualHandler"]

import logging
from typing import TYPE_CHECKING
from unittest.mock import Mock

from coola.equality.handlers.base import BaseEqualityHandler
from coola.utils import is_pandas_available

if is_pandas_available():
    import pandas
else:  # pragma: no cover
    pandas = Mock()

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig

logger = logging.getLogger(__name__)


class PandasSeriesEqualHandler(BaseEqualityHandler):
    r"""Check if the two ``pandas.Series`` are equal.

    This handler returns ``True`` if the two arrays ``pandas.Series``
    equal, otherwise ``False``. This handler is designed to be used
    at the end of the chain of responsibility. This handler does
    not call the next handler.

    Example usage:

    ```pycon
    >>> import pandas
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.handlers import PandasSeriesEqualHandler
    >>> from coola.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> handler = PandasSeriesEqualHandler()
    >>> handler.handle(pandas.Series([1, 2, 3, 4, 5]), pandas.Series([1, 2, 3, 4, 5]), config)
    True
    >>> handler.handle(pandas.Series([1, 2, 3, 4, 5]), pandas.Series([1, 2, 3, 4, 0]), config)
    False

    ```
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def handle(
        self,
        object1: pandas.Series,
        object2: pandas.Series,
        config: EqualityConfig,
    ) -> bool:
        object_equal = self._compare_series(object1, object2, config)
        if config.show_difference and not object_equal:
            logger.info(
                f"pandas.Series have different elements:\n"
                f"object1:\n{object1}\nobject2:\n{object2}"
            )
        return object_equal

    def set_next_handler(self, handler: BaseEqualityHandler) -> None:
        pass  # Do nothing because the next handler is never called.

    def _compare_series(
        self, series1: pandas.Series, series2: pandas.Series, config: EqualityConfig
    ) -> bool:
        r"""Indicate if the two series are equal or not.

        Args:
            series1: Specifies the first series to compare.
            series2: Specifies the second series to compare.
            config: Specifies the equality configuration.

        Returns:
            ``True``if the two series are equal, otherwise ``False``.
        """
        if not config.equal_nan and series1.isna().any():
            return False
        try:
            pandas.testing.assert_series_equal(series1, series2, check_exact=True)
        except AssertionError:
            return False
        return True
