r"""Contain utility functions to manage randomness."""

from __future__ import annotations

__all__ = ["get_random_managers"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coola.random.base import BaseRandomManager


def get_random_managers() -> dict[str, BaseRandomManager]:
    r"""Get the random managers and their default name.

    Returns:
        The mapping between the name and random managers.

    Example usage:

    ```pycon

    >>> from coola.random import get_random_managers
    >>> get_random_managers()
    {'random': RandomRandomManager()...}

    ```
    """
    from coola import random  # Local import to avoid cyclic dependencies

    return (
        random.random_.get_random_managers()
        | random.numpy_.get_random_managers()
        | random.torch_.get_random_managers()
    )
