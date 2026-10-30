r"""Define the base class to manage randomness."""

from __future__ import annotations

__all__ = ["BaseRandomManager"]

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from coola.random.seed import validate_seed

if TYPE_CHECKING:
    from collections.abc import Generator


class BaseRandomManager(ABC):
    r"""Implement the base class to manage randomness.

    Each child class must implement the methods:
    - ``get_rng_state``.
    - ``manual_seed``.
    - ``set_rng_state``.

    The ``seed`` context manager is implemented with these methods.

    A manager follows this contract:

    1. ``get_rng_state`` returns an opaque object that does not alias
        the live generator: drawing a number afterwards does not change
        it.
    2. ``set_rng_state(get_rng_state())`` leaves the draws unchanged.
    3. ``manual_seed`` validates the seed with ``validate_seed`` before
        it touches a generator, and the same seed gives the same draws.
        ``seed`` validates the seed too, so a manager gets the seed
        domain even if its ``manual_seed`` does not check.
    4. A manager is stateless and cheap to build: it is shared by
        registries, equal to another instance of its class and
        hashable (``__eq__`` and ``__hash__``).
    5. Building a manager of an uninstalled library raises an
        ``ImportError``.

    Only the global generators are managed. The generators created by
    the user (``random.Random()``, ``numpy.random.default_rng()``,
    ``torch.Generator()``) are not.

    Example:
        ```pycon
        >>> from coola.random import TorchRandomManager
        >>> manager = TorchRandomManager()
        >>> manager.manual_seed(42)

        ```
    """

    @abstractmethod
    def get_rng_state(self) -> object:
        r"""Get the current RNG state.

        Returns:
            The current RNG state.

        Example:
            ```pycon
            >>> from coola.random import TorchRandomManager
            >>> manager = TorchRandomManager()
            >>> state = manager.get_rng_state()
            >>> state
            {'torch': tensor([...], dtype=torch.uint8), ...}

            ```
        """

    @abstractmethod
    def manual_seed(self, seed: int) -> None:
        r"""Set the seed for generating random numbers.

        Args:
            seed: The desired seed.

        Example:
            ```pycon
            >>> from coola.random import TorchRandomManager
            >>> manager = TorchRandomManager()
            >>> manager.manual_seed(42)

            ```
        """

    @abstractmethod
    def set_rng_state(self, state: Any) -> None:
        r"""Set the RNG state.

        Args:
            state: The new RNG state.

        Example:
            ```pycon
            >>> import torch
            >>> from coola.random import TorchRandomManager
            >>> manager = TorchRandomManager()
            >>> state = manager.get_rng_state()
            >>> manager.set_rng_state(state)

            ```
        """

    @contextmanager
    def seed(self, seed: int) -> Generator[None]:
        r"""Temporarily set a seed and restore the RNG state afterward.

        The state is restored even if an exception is raised inside the
        context.

        The seed is validated with ``validate_seed`` before the state
        is read, so an invalid seed changes nothing.

        Args:
            seed: The seed to use while inside the context manager.

        Yields:
            ``None``. The seed is set while the block runs.

        Raises:
            TypeError: If the seed is not an ``int``.
            ValueError: If the seed is not in the range ``[0, 2**63)``.

        Example:
            ```pycon
            >>> import random
            >>> from coola.random import PythonRandomManager
            >>> manager = PythonRandomManager()
            >>> with manager.seed(42):
            ...     first = random.random()
            ...
            >>> with manager.seed(42):
            ...     second = random.random()
            ...
            >>> first == second
            True

            ```
        """
        validate_seed(seed)
        state = self.get_rng_state()
        try:
            self.manual_seed(seed)
            yield
        finally:
            self.set_rng_state(state)
