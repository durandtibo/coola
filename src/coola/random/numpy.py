r"""Implement a random manager for NumPy."""

from __future__ import annotations

__all__ = ["NumpyRandomManager", "get_numpy_generator"]

from typing import TYPE_CHECKING, Any

from coola.display import InlineDisplayMixin
from coola.random.base import BaseRandomManager
from coola.random.seed import validate_seed
from coola.utils.imports import check_numpy, is_numpy_available
from coola.utils.introspection import is_same_type

if TYPE_CHECKING or is_numpy_available():
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

    The legacy generator only accepts a scalar seed between ``0`` and
    ``2**32 - 1``. A seed below ``2**32`` is given as is. A larger seed
    is split in two 32-bit words and given as an array, so two
    different seeds in ``[0, 2**63)`` never give the same random
    numbers.

    Raises:
        MissingPackageError: If ``numpy`` is not installed.

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
        r"""Set the seed of the NumPy random number generator.

        Args:
            seed: The seed.

        Raises:
            TypeError: If the seed is not an ``int``.
            ValueError: If the seed is not in the range ``[0, 2**63)``.
        """
        validate_seed(seed)
        if seed < 2**32:
            np.random.seed(seed)
        else:  # the legacy generator accepts an array of 32-bit words: no folding
            np.random.seed([seed & 0xFFFFFFFF, seed >> 32])

    def set_rng_state(self, state: tuple[Any, ...]) -> None:
        np.random.set_state(state)


def get_numpy_generator(seed: int) -> np.random.Generator:
    r"""Create a ``numpy.random.Generator`` initialized with a given seed.

    The generator is independent from the global generator of NumPy: it
    is the way to get reproducible random numbers in a thread, where
    ``random_seed`` must not be used.

    Args:
        seed: A random seed.

    Returns:
        A ``numpy.random.Generator`` object.

    Raises:
        TypeError: If the seed is not an ``int``.
        ValueError: If the seed is not in the range ``[0, 2**63)``.
        MissingPackageError: If ``numpy`` is not installed.

    Example:
        ```pycon
        >>> from coola.random import get_numpy_generator
        >>> generator = get_numpy_generator(42)
        >>> generator.random(2)
        array([0.77395605, 0.43887844])
        >>> generator = get_numpy_generator(42)
        >>> generator.random(2)
        array([0.77395605, 0.43887844])

        ```
    """
    check_numpy()
    validate_seed(seed)
    return np.random.default_rng(seed)
