r"""Implement a random manager for the python standard library
``random``."""

from __future__ import annotations

__all__ = ["RandomRandomManager"]

import random
from typing import Any

from coola.random.base import BaseRandomManager


class RandomRandomManager(BaseRandomManager):  # noqa: PLW1641
    r"""Implement a random manager for the python standard library
    ``random``.

    Example:
        ```pycon
        >>> from coola.random import RandomRandomManager
        >>> manager = RandomRandomManager()
        >>> manager.manual_seed(42)

        ```
    """

    def __eq__(self, other: object) -> bool:
        return type(other) is type(self)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def get_rng_state(self) -> tuple[Any, ...]:
        return random.getstate()

    def manual_seed(self, seed: int) -> None:
        random.seed(seed)

    def set_rng_state(self, state: tuple[Any, ...]) -> None:
        random.setstate(state)
