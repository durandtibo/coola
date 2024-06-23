r"""Implement some utility functions for ``numpy.ndarray``s."""

from __future__ import annotations

__all__ = ["to_array"]

from typing import TYPE_CHECKING
from unittest.mock import Mock

from coola.utils.imports import is_numpy_available, is_torch_available

if is_numpy_available():
    import numpy as np
else:
    np = Mock()  # pragma: no cover

if is_torch_available():
    import torch
else:  # pragma: no cover
    torch = Mock()

if TYPE_CHECKING:
    from collections.abc import Sequence


def to_array(data: Sequence | torch.Tensor | np.ndarray) -> np.ndarray:
    r"""Convert the input to a ``numpy.ndarray``.

    Args:
        data: The data to convert to a NumPy array.

    Returns:
        A NumPy array.

    Example usage:

    ```pycon

    >>> from coola.utils.array import to_array
    >>> x = to_array([1, 2, 3, 4, 5])
    >>> x
    array([1, 2, 3, 4, 5])
    >>> import torch
    >>> x = to_array(torch.tensor([1, 2, 3, 4, 5]))
    >>> x
    array([1, 2, 3, 4, 5])

    ```
    """
    if is_torch_available() and torch.is_tensor(data):
        return data.numpy()
    if not isinstance(data, np.ndarray):
        return np.asarray(data)
    return data
