r"""Implement summarizers for pandas objects.

The summarizers show metadata (shape, dtype, column names) instead of
the values by default.
"""

from __future__ import annotations

__all__ = ["PandasDataFrameSummarizer", "PandasSeriesSummarizer"]

from typing import TYPE_CHECKING

from coola.summary.metadata import BaseMetadataSummarizer, format_fields
from coola.utils.imports import check_pandas, is_pandas_available

if TYPE_CHECKING or is_pandas_available():
    import pandas as pd
else:  # pragma: no cover
    from coola.utils.fallback.pandas import pandas as pd


class PandasDataFrameSummarizer(BaseMetadataSummarizer[pd.DataFrame]):
    r"""Implement a summarizer for ``pandas.DataFrame`` objects.

    By default, it displays the metadata (type, shape, columns with their
    dtype) rather than the values, which makes it suitable for logging
    large data frames. At most 10 columns are listed.

    Args:
        show_data: If ``True``, returns ``repr(dataframe)``, displaying
            the values. If ``False`` (default), returns only the
            metadata: ``<class> | shape=<shape> | columns=<name: dtype, ...>``.

    Raises:
        MissingPackageError: If pandas is not installed.

    Example:
        ```pycon
        >>> import pandas as pd
        >>> from coola.summary import SummarizerRegistry, PandasDataFrameSummarizer
        >>> registry = SummarizerRegistry()
        >>> summarizer = PandasDataFrameSummarizer()
        >>> frame = pd.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
        >>> print(summarizer.summarize(frame, registry))
        <class 'pandas.DataFrame'> | shape=(3, 2) | columns=a: int64, b: float64

        ```
    """

    def _check_dependencies(self) -> None:
        check_pandas()

    def _describe(self, data: pd.DataFrame) -> list[str]:
        return [
            f"shape={data.shape}",
            f"columns={format_fields(data.dtypes.items())}",
        ]


class PandasSeriesSummarizer(BaseMetadataSummarizer[pd.Series]):
    r"""Implement a summarizer for ``pandas.Series`` objects.

    By default, it displays the metadata (type, shape, dtype, name)
    rather than the values.

    Args:
        show_data: If ``True``, returns ``repr(series)``, displaying
            the values. If ``False`` (default), returns only the
            metadata: ``<class> | shape=<shape> | dtype=<dtype> | name=<name>``.

    Raises:
        MissingPackageError: If pandas is not installed.

    Example:
        ```pycon
        >>> import pandas as pd
        >>> from coola.summary import SummarizerRegistry, PandasSeriesSummarizer
        >>> registry = SummarizerRegistry()
        >>> summarizer = PandasSeriesSummarizer()
        >>> print(summarizer.summarize(pd.Series([1, 2, 3], name="a"), registry))
        <class 'pandas.Series'> | shape=(3,) | dtype=int64 | name='a'

        ```
    """

    def _check_dependencies(self) -> None:
        check_pandas()

    def _describe(self, data: pd.Series) -> list[str]:
        return [f"shape={data.shape}", f"dtype={data.dtype}", f"name={data.name!r}"]
