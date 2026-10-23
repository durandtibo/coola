r"""Implement a random manager for NumPy."""

from __future__ import annotations

__all__ = ["NumpyRandomManager"]

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from coola.display import InlineDisplayMixin
from coola.random.base import BaseRandomManager
from coola.utils.imports import check_numpy, is_numpy_available
from coola.utils.introspection import is_same_type

if TYPE_CHECKING:
    from collections.abc import Generator

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
    produce the same random numbers, and a negative seed ``-n`` is the
    same as ``2**32 - n``.

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
        np.random.seed(seed % 2**32)

    def set_rng_state(self, state: tuple[Any, ...]) -> None:
        np.random.set_state(state)


@contextmanager
def numpy_seed(seed: int) -> Generator[None]:
    r"""Implement a context manager to manage the NumPy random seed and
    random number generator (RNG) state.

    The context manager sets the specified random seed and
    restores the original RNG state afterward.

    Args:
        seed: The random number generator seed to use while using
            this context manager.

    Example:
        ```pycon
        >>> import numpy
        >>> from coola.random import numpy_seed
        >>> with numpy_seed(42):
        ...     print(numpy.random.randn(2, 4))
        ...
        [[...]]
        >>> with numpy_seed(42):
        ...     print(numpy.random.randn(2, 4))
        ...
        [[...]]

        ```
    """
    with NumpyRandomManager().seed(seed):
        yield
