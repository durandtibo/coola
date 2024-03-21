r"""Implement a random seed setter for NumPy."""

from __future__ import annotations

__all__ = ["NumpyRandomManager"]

from unittest.mock import Mock

from coola.random.base import BaseRandomManager
from coola.utils import check_numpy, is_numpy_available

if is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    np = Mock()


class NumpyRandomManager(BaseRandomManager):
    r"""Implement a random seed setter for the library ``numpy``.

    The seed must be between ``0`` and ``2**32 - 1``, so a modulo
    operator to convert an integer to an integer between ``0`` and
    ``2**32 - 1``.

    Example usage:

    ```pycon

    >>> from coola.random import NumpyRandomManager
    >>> setter = NumpyRandomManager()
    >>> setter.manual_seed(42)

    ```
    """

    def __init__(self) -> None:
        check_numpy()

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def get_rng_state(self) -> dict | tuple:
        return np.random.get_state()

    def manual_seed(self, seed: int) -> None:
        np.random.seed(seed % 2**32)

    def set_rng_state(self, state: dict | tuple) -> None:
        np.random.set_state(state)
