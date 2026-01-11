r"""Implement the tensor summarizer for PyTorch tensors.

This module provides a summarizer that creates compact, informative
string representations of PyTorch tensors, showing metadata like shape,
dtype, and device instead of the full data by default.
"""

from __future__ import annotations

__all__ = ["TensorSummarizer"]

from typing import TYPE_CHECKING

from coola.summary.base import BaseSummarizer
from coola.utils import check_torch, is_torch_available

if TYPE_CHECKING:
    from coola.summary.registry import SummarizerRegistry

if TYPE_CHECKING or is_torch_available():
    import torch
else:  # pragma: no cover
    from coola.utils.fallback.torch import torch


class TensorSummarizer(BaseSummarizer[torch.Tensor]):
    r"""Implement a summarizer for ``torch.Tensor`` objects.

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
        data: torch.Tensor,
        registry: SummarizerRegistry,  # noqa: ARG002
        depth: int = 0,  # noqa: ARG002
        max_depth: int = 1,  # noqa: ARG002
    ) -> str:
        if self._show_data:
            return repr(data)
        return " | ".join(
            [
                f"{type(data)}",
                f"shape={data.shape}",
                f"dtype={data.dtype}",
                f"device={data.device}",
                f"requires_grad={data.requires_grad}",
            ]
        )
