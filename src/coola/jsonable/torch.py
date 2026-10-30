r"""Define the transformers of the PyTorch objects."""

from __future__ import annotations

__all__ = ["get_default_transformers"]

from typing import TYPE_CHECKING, Any

from coola.jsonable.transformers import ConvertTransformer
from coola.utils.imports import is_torch_available

if TYPE_CHECKING or is_torch_available():
    import torch
else:  # pragma: no cover
    from coola.utils.fallback.torch import torch

if TYPE_CHECKING:
    from coola.recursive.base import Transformer


def get_default_transformers() -> dict[type, Transformer[Any]]:
    r"""Get the default transformers of the PyTorch objects.

    A tensor gives the nested lists of ``tolist``, which are converted
    in turn, so complex numbers are rejected.

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
    return {torch.Tensor: ConvertTransformer(lambda tensor: tensor.tolist())}
