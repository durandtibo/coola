r"""Implement a random manager for NumPy."""

from __future__ import annotations

__all__ = ["NumpyRandomManager", "get_random_managers"]

from unittest.mock import Mock

from coola.random.base import BaseRandomManager
from coola.utils import check_numpy, is_numpy_available

if is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    np = Mock()


class NumpyRandomManager(BaseRandomManager):
    r"""Implement a random manager for the library ``numpy``.

    The seed must be between ``0`` and ``2**32 - 1``, so a modulo
    operator to convert an integer to an integer between ``0`` and
    ``2**32 - 1``.

    Example usage:

    ```pycon

    >>> from coola.random import NumpyRandomManager
    >>> manager = NumpyRandomManager()
    >>> manager.manual_seed(42)

    ```
    """

    def __init__(self) -> None:
        check_numpy()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def get_rng_state(self) -> dict | tuple:
        return np.random.get_state()

    def manual_seed(self, seed: int) -> None:
        np.random.seed(seed % 2**32)

    def set_rng_state(self, state: dict | tuple) -> None:
        np.random.set_state(state)


def get_random_managers() -> dict[str, BaseRandomManager]:
    r"""Get the random managers and their default name.

    This function returns an empty dictionary if ``numpy`` is not
    installed.

    Returns:
        The mapping between the name and random managers.

    Example usage:

    ```pycon
    >>> from coola.random.numpy_ import get_random_managers
    >>> get_random_managers()
    {'numpy': NumpyRandomManager()}

    ```
    """
    if not is_numpy_available():
        return {}
    return {"numpy": NumpyRandomManager()}
