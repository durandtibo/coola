r"""Implement the tensor summarizer for PyTorch tensors.

This module provides a summarizer that creates compact, informative
string representations of PyTorch tensors, showing metadata like shape,
dtype, and device instead of the full data by default.
"""

from __future__ import annotations

__all__ = ["TensorSummarizer"]

from typing import TYPE_CHECKING

from coola.summary.metadata import BaseMetadataSummarizer
from coola.utils.imports import check_torch

if TYPE_CHECKING:
    import torch


class TensorSummarizer(BaseMetadataSummarizer["torch.Tensor"]):
    r"""Implement a summarizer for ``torch.Tensor`` objects.

    This summarizer generates compact string representations of PyTorch
    tensors. By default, it displays metadata (type, shape, dtype, device,
    requires_grad) rather than tensor values, making it suitable for logging
    and debugging large tensors. Optionally, it can show the full tensor
    representation.

    Args:
        show_data: If ``True``, returns the default tensor string
            representation (same as ``repr(tensor)``), displaying
            actual values. If ``False`` (default), returns only
            metadata in a compact format:
            ``<class> | shape=<shape> | dtype=<dtype> | device=<device> |
            requires_grad=<requires_grad>``.
            Default: ``False``

    Raises:
        MissingPackageError: If PyTorch is not installed or available.

    Example:
        ```pycon
        >>> import torch
        >>> from coola.summary import SummarizerRegistry, TensorSummarizer
        >>> registry = SummarizerRegistry()

        >>> # Default behavior: show metadata only
        >>> summarizer = TensorSummarizer()
        >>> print(summarizer.summarize(torch.arange(11), registry))  # doctest: +ELLIPSIS
        <class 'torch.Tensor'> | shape=torch.Size([11]) | dtype=torch.int64 | device=cpu | requires_grad=False

        >>> # Works with tensors of any shape and dtype
        >>> print(summarizer.summarize(torch.ones(2, 3, 4), registry))  # doctest: +ELLIPSIS
        <class 'torch.Tensor'> | shape=torch.Size([2, 3, 4]) | dtype=torch.float32 | device=cpu | requires_grad=False

        >>> # Show full tensor data
        >>> summarizer = TensorSummarizer(show_data=True)
        >>> print(summarizer.summarize(torch.arange(11), registry))  # doctest: +ELLIPSIS
        tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

        ```
    """

    def _check_dependencies(self) -> None:
        check_torch()

    def _describe(self, data: torch.Tensor) -> list[str]:
        return [
            f"shape={data.shape}",
            f"dtype={data.dtype}",
            f"device={data.device}",
            f"requires_grad={data.requires_grad}",
        ]
