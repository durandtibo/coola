r"""Implement the NDArray summarizer for NumPy arrays.

This module provides a summarizer that creates compact, informative
string representations of NumPy ndarrays, showing metadata like shape
and dtype instead of the full data by default.
"""

from __future__ import annotations

__all__ = ["NDArraySummarizer"]

from typing import TYPE_CHECKING

from coola.summary.base import BaseSummarizer
from coola.utils import check_numpy, is_numpy_available

if TYPE_CHECKING:
    from coola.summary.registry import SummarizerRegistry

if TYPE_CHECKING or is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    from coola.utils.fallback.numpy import numpy as np


class NDArraySummarizer(BaseSummarizer[np.ndarray]):
    r"""Implement a summarizer for ``numpy.ndarray`` objects.

    This summarizer generates compact string representations of NumPy
    arrays. By default, it displays metadata (type, shape, dtype)
    rather than array values, making it suitable for logging and debugging
    large arrays. Optionally, it can show the full array representation.

    Args:
        show_data: If ``True``, returns the default array string
            representation (same as ``repr(array)``), displaying
            actual values. If ``False`` (default), returns only
            metadata in a compact format:
            ``<class> | shape=<shape> | dtype=<dtype>``.
            Default: ``False``

    Raises:
        RuntimeError: If NumPy is not installed or available.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from coola.summary import SummarizerRegistry, NDArraySummarizer
        >>> registry = SummarizerRegistry()

        >>> # Default behavior: show metadata only
        >>> summarizer = NDArraySummarizer()
        >>> summarizer.summarize(np.arange(11), registry)
        <class 'numpy.ndarray'> | shape=(11,) | dtype=int64

        >>> # Works with arrays of any shape and dtype
        >>> summarizer.summarize(np.ones((2, 3, 4)), registry)
        <class 'numpy.ndarray'> | shape=(2, 3, 4) | dtype=float64

        >>> # Show full array data
        >>> summarizer = NDArraySummarizer(show_data=True)
        >>> summarizer.summarize(np.arange(11), registry)
        array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

        ```
    """

    def __init__(self, show_data: bool = False) -> None:
        check_numpy()
        self._show_data = bool(show_data)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(show_data={self._show_data})"

    def equal(self, other: object) -> bool:
        if type(other) is not type(self):
            return False
        return self._show_data == other._show_data

    def summarize(
        self,
        data: np.ndarray,
        registry: SummarizerRegistry,  # noqa: ARG002
        depth: int = 0,  # noqa: ARG002
        max_depth: int = 1,  # noqa: ARG002
    ) -> str:
        if self._show_data:
            return repr(data)
        return " | ".join([f"{type(data)}", f"shape={data.shape}", f"dtype={data.dtype}"])
