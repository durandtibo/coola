r"""Define the base class to set a random seed."""

from __future__ import annotations

__all__ = ["BaseRandomSeedSetter"]

from abc import ABC, abstractmethod


class BaseRandomSeedSetter(ABC):
    r"""Implement the base class to implement a random seed setter.

    Each child class must implement the method ``manual_seed``.

    Example usage:

    ```pycon

    >>> from coola.random import TorchRandomSeedSetter
    >>> setter = TorchRandomSeedSetter()
    >>> setter.manual_seed(42)

    ```
    """

    @abstractmethod
    def manual_seed(self, seed: int) -> None:
        r"""Set the seed for generating random numbers.

        Args:
            seed: The desired seed.

        Example usage:

        ```pycon

        >>> from coola.random import TorchRandomSeedSetter
        >>> setter = TorchRandomSeedSetter()
        >>> setter.manual_seed(42)

        ```
        """
