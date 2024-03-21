r"""Implement a random seed setter for the python standard library
``random``."""

from __future__ import annotations

__all__ = ["RandomRandomManager"]

import random

from coola.random.base import BaseRandomManager


class RandomRandomManager(BaseRandomManager):
    r"""Implement a random seed setter for the python standard library
    ``random``.

    Example usage:

    ```pycon

    >>> from coola.random import RandomRandomManager
    >>> setter = RandomRandomManager()
    >>> setter.manual_seed(42)

    ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def get_rng_state(self) -> tuple:
        return random.getstate()

    def manual_seed(self, seed: int) -> None:
        random.seed(seed)

    def set_rng_state(self, state: tuple) -> None:
        random.setstate(state)
