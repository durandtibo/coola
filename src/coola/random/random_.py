r"""Implement a random manager for the python standard library
``random``."""

from __future__ import annotations

__all__ = ["RandomRandomManager", "get_random_managers"]

import random

from coola.random.base import BaseRandomManager


class RandomRandomManager(BaseRandomManager):
    r"""Implement a random manager for the python standard library
    ``random``.

    Example usage:

    ```pycon

    >>> from coola.random import RandomRandomManager
    >>> manager = RandomRandomManager()
    >>> manager.manual_seed(42)

    ```
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def get_rng_state(self) -> tuple:
        return random.getstate()

    def manual_seed(self, seed: int) -> None:
        random.seed(seed)

    def set_rng_state(self, state: tuple) -> None:
        random.setstate(state)


def get_random_managers() -> dict[str, BaseRandomManager]:
    r"""Get the random managers and their default name.

    Returns:
        The mapping between the name and random managers.

    Example usage:

    ```pycon

    >>> from coola.random.random_ import get_random_managers
    >>> get_random_managers()
    {'random': RandomRandomManager()}

    ```
    """
    return {"random": RandomRandomManager()}
