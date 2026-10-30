r"""Implement the tensor summarizer for PyTorch tensors.

This module provides a summarizer that creates compact, informative
string representations of PyTorch tensors, showing metadata like shape,
dtype, and device instead of the full data by default.
"""

from __future__ import annotations

__all__ = ["TensorSummarizer"]

from typing import TYPE_CHECKING, Any

from coola.summary.metadata import BaseMetadataSummarizer, Field
from coola.utils.imports import check_torch, is_torch_available

if TYPE_CHECKING or is_torch_available():
    import torch
else:  # pragma: no cover
    from coola.utils.fallback.torch import torch


if TYPE_CHECKING:
    from coola.summary.base import Summarizer


class TensorSummarizer(BaseMetadataSummarizer[torch.Tensor]):
    r"""Implement a summarizer for ``torch.Tensor`` objects.

    This summarizer generates compact string representations of PyTorch
    tensors. By default, it displays metadata (type, shape, dtype, device,
    requires_grad) rather than tensor values, making it suitable for logging
    and debugging large tensors. Optionally, it can show the full tensor
    representation.

    Raises:
        MissingPackageError: If PyTorch is not installed or available.

    Example:
        ```pycon
        >>> import torch
        >>> from coola.summary import TensorSummarizer, SummaryConfig

        >>> # Default behavior: show metadata only
        >>> summarizer = TensorSummarizer()
        >>> print(summarizer.summarize(torch.arange(11), SummaryConfig(), compact=False))  # doctest: +ELLIPSIS
        <class 'torch.Tensor'> | shape=torch.Size([11]) | dtype=torch.int64 | device=cpu | requires_grad=False

        >>> # Works with tensors of any shape and dtype
        >>> print(summarizer.summarize(torch.ones(2, 3, 4), SummaryConfig(), compact=False))  # doctest: +ELLIPSIS
        <class 'torch.Tensor'> | shape=torch.Size([2, 3, 4]) | dtype=torch.float32 | device=cpu | requires_grad=False

        >>> # Show full tensor data
        >>> config = SummaryConfig(show_data=True)
        >>> print(summarizer.summarize(torch.arange(11), config, compact=False))  # doctest: +ELLIPSIS
        tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

        ```
    """

    def _check_dependencies(self) -> None:
        check_torch()

    def _describe(self, data: torch.Tensor) -> list[Field]:
        return [
            f"shape={data.shape}",
            f"dtype={data.dtype}",
            f"device={data.device}",
            f"requires_grad={data.requires_grad}",
        ]


def get_default_summarizers() -> dict[type, Summarizer[Any]]:
    r"""Get the default summarizers of the PyTorch tensors.

    Returns:
        The summarizers of the PyTorch tensors and their associated types.

    Notes:
        This function is called by the default registry the first time it
        needs the summarizers of the PyTorch tensors.
    """
    if not is_torch_available():
        return {}

    return {torch.Tensor: TensorSummarizer()}
