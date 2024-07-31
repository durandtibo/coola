r"""Implement a random manager for PyTorch."""

from __future__ import annotations

__all__ = ["TorchRandomManager", "get_random_managers", "torch_seed"]

from contextlib import contextmanager
from typing import TYPE_CHECKING
from unittest.mock import Mock

from coola.random.base import BaseRandomManager
from coola.utils import check_torch, is_torch_available

if TYPE_CHECKING:
    from collections.abc import Generator

if is_torch_available():
    import torch
else:  # pragma: no cover
    torch = Mock()


class TorchRandomManager(BaseRandomManager):
    r"""Implements a random number generator for the library ``torch``.

    Example usage:

    ```pycon

    >>> from coola.random import TorchRandomManager
    >>> manager = TorchRandomManager()
    >>> manager.manual_seed(42)

    ```
    """

    def __init__(self) -> None:
        check_torch()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def get_rng_state(self) -> dict:
        return {
            "torch": torch.get_rng_state(),
            "torch.cuda": torch.cuda.get_rng_state_all(),
        }

    def manual_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def set_rng_state(self, state: dict) -> None:
        torch.set_rng_state(state["torch"])
        torch.cuda.set_rng_state_all(state["torch.cuda"])


def get_random_managers() -> dict[str, BaseRandomManager]:
    r"""Get the random managers and their default name.

    This function returns an empty dictionary if ``torch`` is not
    installed.

    Returns:
        The mapping between the name and random managers.

    Example usage:

    ```pycon

    >>> from coola.random.torch_ import get_random_managers
    >>> get_random_managers()
    {'torch': TorchRandomManager()}

    ```
    """
    if not is_torch_available():
        return {}
    return {"torch": TorchRandomManager()}


@contextmanager
def torch_seed(seed: int) -> Generator[None]:
    r"""Implement a context manager to manage the PyTorch random seed and
    random number generator (RNG) state.

    The context manager sets the specified random seed and
    restores the original RNG state afterward.

    Args:
        seed: The random number generator seed to use while using
            this context manager.

    Example usage:

    ```pycon

    >>> import torch
    >>> from coola.random import torch_seed
    >>> with torch_seed(42):
    ...     print(torch.randn(2, 4))
    ...
    tensor([[...]])
    >>> with torch_seed(42):
    ...     print(torch.randn(2, 4))
    ...
    tensor([[...]])

    ```
    """
    manager = TorchRandomManager()
    state = manager.get_rng_state()
    try:
        manager.manual_seed(seed)
        yield
    finally:
        manager.set_rng_state(state)
