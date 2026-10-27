r"""Implement a random manager for NumPy."""

from __future__ import annotations

__all__ = ["NumpyRandomManager"]

from typing import Any

from coola.display import InlineDisplayMixin
from coola.random.base import BaseRandomManager
from coola.random.seed import validate_seed
from coola.utils.imports import check_numpy, is_numpy_available
from coola.utils.introspection import is_same_type

if is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    from coola.utils.fallback.numpy import numpy as np


class NumpyRandomManager(InlineDisplayMixin, BaseRandomManager):
    r"""Implement a random manager for the library ``numpy``.

    Only the legacy global generator of NumPy (``numpy.random.seed``,
    ``numpy.random.get_state`` and ``numpy.random.set_state``) is
    managed. The generators created with ``numpy.random.default_rng()``
    are independent objects: they are not seeded, and their state is
    not saved nor restored. Seed them explicitly, e.g.
    ``numpy.random.default_rng(seed)``.

    The legacy generator only accepts seeds between ``0`` and
    ``2**32 - 1``, so the seed is folded into this range with
    ``seed % 2**32``. Consequently, the seeds ``1`` and ``2**32 + 1``
    produce the same random numbers. The seed is validated first: a
    negative seed raises a ``ValueError``.

    Example:
        ```pycon
        >>> from coola.random import NumpyRandomManager
        >>> manager = NumpyRandomManager()
        >>> manager.manual_seed(42)

        ```
    """

    def __init__(self) -> None:
        check_numpy()

    def __eq__(self, other: object) -> bool:
        return is_same_type(other, self)

    def __hash__(self) -> int:
        return hash(type(self))

    def get_rng_state(self) -> tuple[Any, ...]:
        return np.random.get_state()

    def manual_seed(self, seed: int) -> None:
        validate_seed(seed)
        np.random.seed(seed % 2**32)

    def set_rng_state(self, state: tuple[Any, ...]) -> None:
        np.random.set_state(state)
