r"""Implement a random seed setter for PyTorch."""

from __future__ import annotations

__all__ = ["TorchRandomSeedSetter"]

from unittest.mock import Mock

from coola.random.base import BaseRandomSeedSetter
from coola.utils import check_torch, is_torch_available

if is_torch_available():
    import torch
else:  # pragma: no cover
    torch = Mock()


class TorchRandomSeedSetter(BaseRandomSeedSetter):
    r"""Implements a random seed setter for the library ``torch``.

    Example usage:

    ```pycon

    >>> from coola.random import TorchRandomSeedSetter
    >>> setter = TorchRandomSeedSetter()
    >>> setter.manual_seed(42)

    ```
    """

    def __init__(self) -> None:
        check_torch()

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def manual_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
