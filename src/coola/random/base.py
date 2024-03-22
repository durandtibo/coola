r"""Define the base class to manage randomness."""

from __future__ import annotations

__all__ = ["BaseRandomManager"]

from abc import ABC, abstractmethod
from typing import Any


class BaseRandomManager(ABC):
    r"""Implement the base class to manage randomness.

    Each child class must implement the methods:
    - ``get_rng_state``.
    - ``manual_seed``.
    - ``set_rng_state``.

    Example usage:

    ```pycon

    >>> from coola.random import TorchRandomManager
    >>> manager = TorchRandomManager()
    >>> manager.manual_seed(42)

    ```
    """

    @abstractmethod
    def get_rng_state(self) -> Any:
        r"""Get the current RNG state.

        Returns:
            The current RNG state.

        Example usage:

        ```pycon

        >>> from coola.random import TorchRandomManager
        >>> manager = TorchRandomManager()
        >>> state = manager.get_rng_state()
        >>> state
        {'torch': tensor([...], dtype=torch.uint8), 'torch.cuda': ...}

        ```
        """

    @abstractmethod
    def manual_seed(self, seed: int) -> None:
        r"""Set the seed for generating random numbers.

        Args:
            seed: The desired seed.

        Example usage:

        ```pycon

        >>> from coola.random import TorchRandomManager
        >>> manager = TorchRandomManager()
        >>> manager.manual_seed(42)

        ```
        """

    @abstractmethod
    def set_rng_state(self, state: Any) -> None:
        r"""Set the RNG state.

        Args:
            state: The new RNG state.

        Example usage:

        ```pycon

        >>> import torch
        >>> from coola.random import TorchRandomManager
        >>> manager = TorchRandomManager()
        >>> state = manager.get_rng_state()
        >>> manager.set_rng_state(state)

        ```
        """
