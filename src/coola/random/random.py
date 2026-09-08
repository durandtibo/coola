r"""Implement a random manager for the python standard library
``random``."""

from __future__ import annotations

__all__ = ["RandomRandomManager"]

import random
from typing import Any

from coola.display import InlineDisplayMixin
from coola.random.base import BaseRandomManager


class RandomRandomManager(InlineDisplayMixin, BaseRandomManager):  # noqa: PLW1641
    r"""Implement a random manager for the python standard library
    ``random``.

    Example:
        ```pycon
        >>> from coola.random import RandomRandomManager
        >>> manager = RandomRandomManager()
        >>> manager.manual_seed(42)

        ```
    """

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {}

    def __eq__(self, other: object) -> bool:
        return type(other) is type(self)


    def get_rng_state(self) -> tuple[Any, ...]:
        return random.getstate()

    def manual_seed(self, seed: int) -> None:
        random.seed(seed)

    def set_rng_state(self, state: tuple[Any, ...]) -> None:
        random.setstate(state)
