r"""Define the transformers of the PyTorch objects."""

from __future__ import annotations

__all__ = ["TensorTransformer", "get_default_transformers"]

from typing import TYPE_CHECKING, Any

from coola.display import InlineDisplayMixin
from coola.jsonable.transformers import ConvertTransformer
from coola.recursive import BaseTransformer
from coola.utils.imports import is_torch_available

if TYPE_CHECKING or is_torch_available():
    import torch
else:  # pragma: no cover
    from coola.utils.fallback.torch import torch

if TYPE_CHECKING:
    from coola.recursive.base import Expansion, Transformer


class TensorTransformer(InlineDisplayMixin, BaseTransformer["torch.Tensor"]):
    r"""Transformer for PyTorch tensors that rebuilds nested lists.

    The tensor gives the nested lists of ``tolist``. The elements of a
    real tensor are Python ``bool``, ``int`` and ``float``, so the list
    is already JSON data and the engine does not visit its elements. A
    complex tensor gives a list that is converted in turn, so the
    complex numbers are reported.

    Example:
        ```pycon
        >>> import torch
        >>> from coola.jsonable.torch import TensorTransformer
        >>> children, rebuild = TensorTransformer().expand(torch.arange(3))
        >>> children
        ()
        >>> rebuild([])
        [0, 1, 2]

        ```
    """

    def __init__(self) -> None:
        self._convert = ConvertTransformer(lambda tensor: tensor.tolist())

    def expand(self, data: torch.Tensor) -> Expansion | None:
        if data.is_complex():
            return self._convert.expand(data)
        return (), lambda _: data.tolist()


def get_default_transformers() -> dict[type, Transformer[Any]]:
    r"""Get the default transformers of the PyTorch objects.

    A tensor gives the nested lists of ``tolist``; the elements of a real
    tensor are not visited again (see ``TensorTransformer``) and the
    complex numbers are rejected.

    Returns:
        The transformers of the PyTorch objects and their associated
            types.

    Notes:
        This function is called by the default registry the first time it
        needs the transformers of the PyTorch objects.

    Example:
        ```pycon
        >>> from coola.jsonable.torch import get_default_transformers
        >>> [t.__name__ for t in get_default_transformers()]
        ['Tensor']

        ```
    """
    if not is_torch_available():
        return {}
    return {torch.Tensor: TensorTransformer()}
