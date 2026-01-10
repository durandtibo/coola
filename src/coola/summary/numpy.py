r"""Implement the tensor summarizer for PyTorch tensors.

This module provides a summarizer that creates compact, informative
string representations of PyTorch tensors, showing metadata like shape,
dtype, and device instead of the full data by default.
"""

from __future__ import annotations

__all__ = ["NDArraySummarizer"]

from typing import TYPE_CHECKING

from coola.summary.base import BaseSummarizer
from coola.utils import check_torch, is_numpy_available

if TYPE_CHECKING:
    from coola.summary.registry import SummarizerRegistry

if TYPE_CHECKING or is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    from coola.utils.fallback.numpy import numpy as np


class NDArraySummarizer(BaseSummarizer[np.ndarray]):
    r"""Implement a summarizer for ``torch.NDArray`` objects.

    This summarizer generates compact string representations of PyTorch
    tensors. By default, it displays metadata (type, shape, dtype, device)
    rather than tensor values, making it suitable for logging and debugging
    large tensors. Optionally, it can show the full tensor representation.

    Args:
        show_data: If ``True``, returns the default tensor string
            representation (same as ``repr(tensor)``), displaying
            actual values. If ``False`` (default), returns only
            metadata in a compact format:
            ``<class> | shape=<shape> | dtype=<dtype> | device=<device>``.
            Default: ``False``

    Raises:
        RuntimeError: If PyTorch is not installed or available.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from coola.summary import SummarizerRegistry, NDArraySummarizer
        >>> registry = SummarizerRegistry()

        >>> # Default behavior: show metadata only
        >>> summarizer = NDArraySummarizer()
        >>> summarizer.summarize(registry, np.arange(11))
        <class 'numpy.ndarray'> | shape=(11,) | dtype=int64

        >>> # Works with tensors of any shape and dtype
        >>> summarizer.summarize(registry, np.ones((2, 3, 4)))
        <class 'numpy.ndarray'> | shape=(2, 3, 4) | dtype=float64

        >>> # Show full tensor data
        >>> summarizer = NDArraySummarizer(show_data=True)
        >>> summarizer.summarize(registry, np.arange(11))
        array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

        ```
    """

    def __init__(self, show_data: bool = False) -> None:
        check_torch()
        self._show_data = bool(show_data)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(show_data={self._show_data})"

    def equal(self, other: object) -> bool:
        if type(other) is not type(self):
            return False
        return self._show_data == other._show_data

    def summarize(
        self,
        registry: SummarizerRegistry,  # noqa: ARG002
        data: np.ndarray,
        depth: int = 0,  # noqa: ARG002
        max_depth: int = 1,  # noqa: ARG002
    ) -> str:
        if self._show_data:
            return repr(data)
        return " | ".join([f"{type(data)}", f"shape={data.shape}", f"dtype={data.dtype}"])
