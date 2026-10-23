r"""Implement handlers for ``pandas.DataFrame``s and
``pandas.Series``s."""

from __future__ import annotations

__all__ = [
    "PandasDataFrameEqualHandler",
    "PandasExtensionArrayEqualHandler",
    "PandasIndexEqualHandler",
    "PandasSeriesEqualHandler",
]

import logging
from typing import TYPE_CHECKING, Any, cast

from coola.equality.handler.base import BaseEqualityHandler
from coola.equality.handler.format import format_value_difference
from coola.equality.handler.mixin import HandlerEqualityMixin
from coola.equality.handler.utils import report_difference

if TYPE_CHECKING:
    import pandas as pd

    from coola.equality.config import EqualityConfig

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig

logger: logging.Logger = logging.getLogger(__name__)


class PandasDataFrameEqualHandler(HandlerEqualityMixin, BaseEqualityHandler):
    r"""Check if the two ``pandas.DataFrame``s are equal.

    This handler returns ``True`` if the two ``pandas.DataFrame``s are
    equal, otherwise ``False``. This handler is designed to be used
    at the end of the chain of responsibility. This handler does
    not call the next handler.

    Example:
        ```pycon
        >>> import pandas
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import PandasDataFrameEqualHandler
        >>> config = EqualityConfig()
        >>> handler = PandasDataFrameEqualHandler()
        >>> handler.handle(
        ...     pandas.DataFrame({"col": [1, 2, 3]}),
        ...     pandas.DataFrame({"col": [1, 2, 3]}),
        ...     config,
        ... )
        True
        >>> handler.handle(
        ...     pandas.DataFrame({"col": [1, 2, 3]}),
        ...     pandas.DataFrame({"col": [1, 2, 4]}),
        ...     config,
        ... )
        False

        ```
    """

    def handle(
        self,
        actual: pd.DataFrame,
        expected: pd.DataFrame,
        config: EqualityConfig,
    ) -> bool:
        object_equal = frame_equal(actual, expected, config)
        if config.show_difference and not object_equal:
            report_difference(
                logger,
                config,
                format_value_difference(actual=actual, expected=expected, name="pandas.DataFrames"),
            )
        return object_equal


class PandasSeriesEqualHandler(HandlerEqualityMixin, BaseEqualityHandler):
    r"""Check if the two ``pandas.Series``es are equal.

    This handler returns ``True`` if the two ``pandas.Series``es are
    equal, otherwise ``False``. This handler is designed to be used
    at the end of the chain of responsibility. This handler does
    not call the next handler.

    Example:
        ```pycon
        >>> import pandas
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import PandasSeriesEqualHandler
        >>> config = EqualityConfig()
        >>> handler = PandasSeriesEqualHandler()
        >>> handler.handle(pandas.Series([1, 2, 3]), pandas.Series([1, 2, 3]), config)
        True
        >>> handler.handle(pandas.Series([1, 2, 3]), pandas.Series([1, 2, 4]), config)
        False

        ```
    """

    def handle(
        self,
        actual: pd.Series,
        expected: pd.Series,
        config: EqualityConfig,
    ) -> bool:
        object_equal = series_equal(actual, expected, config)
        if config.show_difference and not object_equal:
            report_difference(
                logger,
                config,
                format_value_difference(actual=actual, expected=expected, name="pandas.Series"),
            )
        return object_equal


class PandasIndexEqualHandler(HandlerEqualityMixin, BaseEqualityHandler):
    r"""Check if the two ``pandas.Index``es are equal.

    This handler returns ``True`` if the two ``pandas.Index``es are
    equal (values, dtype and names), otherwise ``False``. This handler
    is designed to be used at the end of the chain of responsibility.
    This handler does not call the next handler.

    Example:
        ```pycon
        >>> import pandas as pd
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import PandasIndexEqualHandler
        >>> config = EqualityConfig()
        >>> handler = PandasIndexEqualHandler()
        >>> handler.handle(pd.Index([1, 2, 3]), pd.Index([1, 2, 3]), config)
        True
        >>> handler.handle(pd.Index([1, 2, 3]), pd.Index([1, 2, 4]), config)
        False

        ```
    """

    def handle(
        self,
        actual: pd.Index,
        expected: pd.Index,
        config: EqualityConfig,
    ) -> bool:
        object_equal = index_equal(actual, expected, config)
        if config.show_difference and not object_equal:
            report_difference(
                logger,
                config,
                format_value_difference(actual=actual, expected=expected, name="pandas.Index"),
            )
        return object_equal


class PandasExtensionArrayEqualHandler(HandlerEqualityMixin, BaseEqualityHandler):
    r"""Check if the two pandas extension arrays are equal.

    The extension arrays include ``pandas.Categorical`` and the arrays
    behind the nullable, string, datetime-with-timezone, interval and
    period dtypes. This handler returns ``True`` if the two arrays are
    equal (values and dtype), otherwise ``False``. This handler is
    designed to be used at the end of the chain of responsibility. This
    handler does not call the next handler.

    Example:
        ```pycon
        >>> import pandas as pd
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import PandasExtensionArrayEqualHandler
        >>> config = EqualityConfig()
        >>> handler = PandasExtensionArrayEqualHandler()
        >>> handler.handle(pd.Categorical(["a", "b"]), pd.Categorical(["a", "b"]), config)
        True
        >>> handler.handle(pd.Categorical(["a", "b"]), pd.Categorical(["a", "c"]), config)
        False

        ```
    """

    def handle(
        self,
        actual: Any,
        expected: Any,
        config: EqualityConfig,
    ) -> bool:
        object_equal = extension_array_equal(actual, expected, config)
        if config.show_difference and not object_equal:
            report_difference(
                logger,
                config,
                format_value_difference(actual=actual, expected=expected, name="pandas arrays"),
            )
        return object_equal


def index_equal(index1: pd.Index, index2: pd.Index, config: EqualityConfig) -> bool:
    r"""Indicate if the two indexes are equal or not.

    Args:
        index1: The first index to compare.
        index2: The second index to compare.
        config: The equality configuration.

    Returns:
        ``True`` if the two indexes are equal, otherwise ``False``.

    Example:
        ```pycon
        >>> import pandas as pd
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler.pandas import index_equal
        >>> config = EqualityConfig()
        >>> index_equal(pd.Index([1, 2, 3]), pd.Index([1, 2, 3]), config)
        True
        >>> index_equal(pd.Index([1, 2, 3]), pd.Index([1, 2, 4]), config)
        False

        ```
    """
    if index1.shape != index2.shape:  # cheap check before scanning for NaN
        return False
    if not config.equal_nan and _index_has_nan(index1):
        return False
    import pandas as pd  # noqa: PLC0415

    try:
        cast("Any", pd.testing).assert_index_equal(
            index1,
            index2,
            exact=True,
            check_exact=config.atol == 0 and config.rtol == 0,
            atol=config.atol,
            rtol=config.rtol,
        )
    except AssertionError:
        return False
    return True


def _index_has_nan(index: pd.Index) -> bool:
    import pandas as pd  # noqa: PLC0415

    if isinstance(index, pd.MultiIndex):  # ``isna`` is not defined for MultiIndex
        return bool(cast("Any", index).to_frame(index=False).isna().to_numpy().any())
    return bool(cast("Any", index).isna().any())


def extension_array_equal(array1: Any, array2: Any, config: EqualityConfig) -> bool:
    r"""Indicate if the two pandas extension arrays are equal or not.

    Args:
        array1: The first array to compare.
        array2: The second array to compare.
        config: The equality configuration.

    Returns:
        ``True`` if the two arrays are equal, otherwise ``False``.

    Example:
        ```pycon
        >>> import pandas as pd
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler.pandas import extension_array_equal
        >>> config = EqualityConfig()
        >>> extension_array_equal(pd.array([1, 2], dtype="Int64"), pd.array([1, 2], dtype="Int64"), config)
        True
        >>> extension_array_equal(pd.array([1, 2], dtype="Int64"), pd.array([1, 3], dtype="Int64"), config)
        False

        ```
    """
    if array1.shape != array2.shape:  # cheap check before scanning for NaN
        return False
    if not config.equal_nan and array1.isna().any():
        return False
    import pandas as pd  # noqa: PLC0415

    try:
        cast("Any", pd.testing).assert_extension_array_equal(
            array1,
            array2,
            check_exact=config.atol == 0 and config.rtol == 0,
            atol=config.atol,
            rtol=config.rtol,
        )
    except AssertionError:
        return False
    return True


def frame_equal(df1: pd.DataFrame, df2: pd.DataFrame, config: EqualityConfig) -> bool:
    r"""Indicate if the two DataFrames are equal or not.

    Args:
        df1: The first DataFrame to compare.
        df2: The second DataFrame to compare.
        config: The equality configuration.

    Returns:
        ``True`` if the two DataFrames are equal, otherwise ``False``.

    Example:
        ```pycon
        >>> import pandas
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler.pandas import frame_equal
        >>> config = EqualityConfig()
        >>> frame_equal(
        ...     pandas.DataFrame({"col": [1, 2, 3]}), pandas.DataFrame({"col": [1, 2, 3]}), config
        ... )
        True
        >>> frame_equal(
        ...     pandas.DataFrame({"col": [1, 2, 3]}), pandas.DataFrame({"col": [1, 2, 4]}), config
        ... )
        False

        ```
    """
    if df1.shape != df2.shape:  # cheap check before scanning for NaN
        return False
    if not config.equal_nan and cast("Any", df1).isna().any().any():
        return False
    import pandas as pd  # noqa: PLC0415

    try:
        cast("Any", pd.testing).assert_frame_equal(
            df1,
            df2,
            check_exact=config.atol == 0 and config.rtol == 0,
            check_index_type=True,
            atol=config.atol,
            rtol=config.rtol,
        )
    except AssertionError:
        return False
    return True


def series_equal(series1: pd.Series, series2: pd.Series, config: EqualityConfig) -> bool:
    r"""Indicate if the two series are equal or not.

    Args:
        series1: The first series to compare.
        series2: The second series to compare.
        config: The equality configuration.

    Returns:
        ``True`` if the two series are equal, otherwise ``False``.

    Example:
        ```pycon
        >>> import pandas
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler.pandas import series_equal
        >>> config = EqualityConfig()
        >>> series_equal(pandas.Series([1, 2, 3]), pandas.Series([1, 2, 3]), config)
        True
        >>> series_equal(pandas.Series([1, 2, 3]), pandas.Series([1, 2, 4]), config)
        False

        ```
    """
    if series1.shape != series2.shape:  # cheap check before scanning for NaN
        return False
    if not config.equal_nan and cast("Any", series1).isna().any():
        return False
    import pandas as pd  # noqa: PLC0415

    try:
        cast("Any", pd.testing).assert_series_equal(
            series1,
            series2,
            check_exact=config.atol == 0 and config.rtol == 0,
            check_index_type=True,
            atol=config.atol,
            rtol=config.rtol,
        )
    except AssertionError:
        return False
    return True
