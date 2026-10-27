r"""Implement the NDArray summarizer for NumPy arrays.

This module provides a summarizer that creates compact, informative
string representations of NumPy ndarrays, showing metadata like shape
and dtype instead of the full data by default.
"""

from __future__ import annotations

__all__ = ["NDArraySummarizer"]

from typing import TYPE_CHECKING, Any

from coola.summary.metadata import BaseMetadataSummarizer, Field
from coola.utils.imports import check_numpy, is_numpy_available

if TYPE_CHECKING or is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    from coola.utils.fallback.numpy import numpy as np


if TYPE_CHECKING:
    from coola.summary.base import Summarizer


class NDArraySummarizer(BaseMetadataSummarizer[np.ndarray]):
    r"""Implement a summarizer for ``numpy.ndarray`` objects.

    This summarizer generates compact string representations of NumPy
    arrays. By default, it displays metadata (type, shape, dtype)
    rather than array values, making it suitable for logging and debugging
    large arrays. Optionally, it can show the full array representation.

    Raises:
        MissingPackageError: If NumPy is not installed or available.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from coola.summary import NDArraySummarizer, SummaryConfig

        >>> # Default behavior: show metadata only
        >>> summarizer = NDArraySummarizer()
        >>> print(summarizer.summarize(np.arange(11), SummaryConfig(), compact=False))
        <class 'numpy.ndarray'> | shape=(11,) | dtype=int64

        >>> # Works with arrays of any shape and dtype
        >>> print(summarizer.summarize(np.ones((2, 3, 4)), SummaryConfig(), compact=False))
        <class 'numpy.ndarray'> | shape=(2, 3, 4) | dtype=float64

        >>> # Show full array data
        >>> config = SummaryConfig(show_data=True)
        >>> print(summarizer.summarize(np.arange(11), config, compact=False))
        array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

        ```
    """

    def _check_dependencies(self) -> None:
        check_numpy()

    def _describe(self, data: np.ndarray) -> list[Field]:
        fields = [f"shape={data.shape}", f"dtype={data.dtype}"]
        if isinstance(data, np.ma.MaskedArray):
            # The mask is part of the value: two arrays with a different mask are different.
            fields.append(f"num_masked={int(np.ma.getmaskarray(data).sum())}")
        return fields


def get_default_summarizers() -> dict[type, Summarizer[Any]]:
    r"""Get the default summarizers of the NumPy arrays.

    Returns:
        The summarizers of the NumPy arrays and their associated types.

    Notes:
        This function is called by the default registry the first time it
        needs the summarizers of the NumPy arrays.
    """
    if not is_numpy_available():
        return {}

    return {np.ndarray: NDArraySummarizer()}
