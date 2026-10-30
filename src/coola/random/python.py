r"""Implement a random manager for the python standard library
``random``."""

from __future__ import annotations

__all__ = ["PythonRandomManager"]

import random
from typing import Any

from coola.display import InlineDisplayMixin
from coola.random.base import BaseRandomManager
from coola.random.seed import validate_seed
from coola.utils.introspection import is_same_type


class PythonRandomManager(InlineDisplayMixin, BaseRandomManager):
    r"""Implement a random manager for the python standard library
    ``random``.

    Only the global generator of the module (``random.seed``,
    ``random.getstate`` and ``random.setstate``) is managed. The
    generators created with ``random.Random()`` are independent objects:
    they are not seeded, and their state is not saved nor restored.

    Example:
        ```pycon
        >>> from coola.random import PythonRandomManager
        >>> manager = PythonRandomManager()
        >>> manager.manual_seed(42)

        ```
    """

    def __eq__(self, other: object) -> bool:
        r"""Indicate if two managers are equal.

        Args:
            other: The other object.

        Returns:
            ``True`` if ``other`` has the same type, otherwise
                ``False``.
        """
        return is_same_type(other, self)

    def __hash__(self) -> int:
        r"""Return the hash of the manager.

        Returns:
            The hash of the type of the manager.
        """
        return hash(type(self))

    def get_rng_state(self) -> tuple[Any, ...]:
        r"""Get the state of the Python random number generator.

        Returns:
            The state, as returned by ``random``.
        """
        return random.getstate()

    def manual_seed(self, seed: int) -> None:
        r"""Set the seed of the Python random number generator.

        Args:
            seed: The seed.

        Raises:
            TypeError: If the seed is not an ``int``.
            ValueError: If the seed is not in the range ``[0, 2**63)``.
        """
        validate_seed(seed)
        random.seed(seed)

    def set_rng_state(self, state: tuple[Any, ...]) -> None:
        r"""Restore the state of the Python random number generator.

        Args:
            state: The state returned by ``get_rng_state``.
        """
        random.setstate(state)
