r"""Implement a random seed setter for the python standard library
``random``."""

from __future__ import annotations

__all__ = ["RandomRandomSeedSetter"]

import random

from coola.random.base import BaseRandomSeedSetter


class RandomRandomSeedSetter(BaseRandomSeedSetter):
    r"""Implement a random seed setter for the python standard library
    ``random``.

    Example usage:

    ```pycon

    >>> from coola.random import RandomRandomSeedSetter
    >>> setter = RandomRandomSeedSetter()
    >>> setter.manual_seed(42)

    ```
    """

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def manual_seed(self, seed: int) -> None:
        random.seed(seed)
