r"""Implement summarizers for pandas objects.

The summarizers show metadata (shape, dtype, column names) instead of
the values by default.
"""

from __future__ import annotations

__all__ = ["PandasDataFrameSummarizer", "PandasSeriesSummarizer"]

from typing import TYPE_CHECKING, Any

from coola.summary.metadata import BaseMetadataSummarizer, Field
from coola.utils.imports import check_pandas, is_pandas_available

if TYPE_CHECKING or is_pandas_available():
    import pandas as pd
else:  # pragma: no cover
    from coola.utils.fallback.pandas import pandas as pd


if TYPE_CHECKING:
    from coola.summary.base import Summarizer


class PandasDataFrameSummarizer(BaseMetadataSummarizer[pd.DataFrame]):
    r"""Implement a summarizer for ``pandas.DataFrame`` objects.

    By default, it displays the metadata (type, shape, columns with their
    dtype) rather than the values, which makes it suitable for logging
    large data frames. At most `max_items` columns are listed.

    Raises:
        MissingPackageError: If pandas is not installed.

    Example:
        ```pycon
        >>> import pandas as pd
        >>> from coola.summary import PandasDataFrameSummarizer, SummaryConfig
        >>> summarizer = PandasDataFrameSummarizer()
        >>> frame = pd.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
        >>> print(summarizer.summarize(frame, SummaryConfig(), compact=False))
        <class 'pandas.DataFrame'> | shape=(3, 2) | columns=a: int64, b: float64

        ```
    """

    def _check_dependencies(self) -> None:
        r"""Check that ``pandas`` is installed.

        Raises:
            MissingPackageError: If ``pandas`` is not installed.
        """
        check_pandas()

    def _describe(self, data: pd.DataFrame) -> list[Field]:
        r"""Describe the metadata of a pandas data frame.

        Args:
            data: The data to describe.

        Returns:
            The shape and the dtype of each column.
        """
        return [
            f"shape={data.shape}",
            ("columns", data.dtypes.items()),
        ]


class PandasSeriesSummarizer(BaseMetadataSummarizer[pd.Series]):
    r"""Implement a summarizer for ``pandas.Series`` objects.

    By default, it displays the metadata (type, shape, dtype, name)
    rather than the values.

    Raises:
        MissingPackageError: If pandas is not installed.

    Example:
        ```pycon
        >>> import pandas as pd
        >>> from coola.summary import PandasSeriesSummarizer, SummaryConfig
        >>> summarizer = PandasSeriesSummarizer()
        >>> print(summarizer.summarize(pd.Series([1, 2, 3], name="a"), SummaryConfig(), compact=False))
        <class 'pandas.Series'> | shape=(3,) | dtype=int64 | name='a'

        ```
    """

    def _check_dependencies(self) -> None:
        r"""Check that ``pandas`` is installed.

        Raises:
            MissingPackageError: If ``pandas`` is not installed.
        """
        check_pandas()

    def _describe(self, data: pd.Series) -> list[Field]:
        r"""Describe the metadata of a pandas series.

        Args:
            data: The data to describe.

        Returns:
            The shape, the dtype and the name.
        """
        return [f"shape={data.shape}", f"dtype={data.dtype}", f"name={data.name!r}"]


def get_default_summarizers() -> dict[type, Summarizer[Any]]:
    r"""Get the default summarizers of the pandas objects.

    Returns:
        The summarizers of the pandas objects and their associated types.

    Notes:
        This function is called by the default registry the first time it
        needs the summarizers of the pandas objects.
    """
    if not is_pandas_available():
        return {}

    return {pd.DataFrame: PandasDataFrameSummarizer(), pd.Series: PandasSeriesSummarizer()}
