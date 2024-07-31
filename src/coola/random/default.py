r"""Implement the default random manager."""

from __future__ import annotations

__all__ = ["RandomManager", "random_seed", "register_random_managers"]

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ClassVar

from coola.random.base import BaseRandomManager
from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping

if TYPE_CHECKING:
    from collections.abc import Generator


class RandomManager(BaseRandomManager):
    r"""Implement the default random manager.

    By default, it is initialized with the following random managers:

    - ``'random'``: ``RandomRandomManager``
    - ``'numpy'``: ``NumpyRandomManager`` if ``numpy`` is available
    - ``'torch'``: ``TorchRandomManager`` if ``torch`` is available

    Example usage:

    ```pycon

    >>> from coola.random import RandomManager
    >>> manager = RandomManager()
    >>> manager.manual_seed(42)

    ```
    """

    registry: ClassVar[dict[str, BaseRandomManager]] = {}

    def __repr__(self) -> str:
        managers = dict(self.registry.items())
        return f"{self.__class__.__qualname__}(\n  {repr_indent(repr_mapping(managers))}\n)"

    def __str__(self) -> str:
        managers = dict(self.registry.items())
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_mapping(managers))}\n)"

    def get_rng_state(self) -> dict[str, Any]:
        return {key: value.get_rng_state() for key, value in self.registry.items()}

    def manual_seed(self, seed: int) -> None:
        for value in self.registry.values():
            value.manual_seed(seed)

    def set_rng_state(self, state: dict[str, Any]) -> None:
        for key, value in state.items():
            self.registry[key].set_rng_state(value)

    @classmethod
    def add_manager(cls, name: str, manager: BaseRandomManager, exist_ok: bool = False) -> None:
        r"""Add a random manager for a given name.

        Args:
            name: The name for the random manager to add.
            manager: The random manager to add.
            exist_ok: If ``False``, ``ValueError`` is raised if the
                name already exists. This parameter should be set to
                ``True`` to overwrite the manager for a name.

        Raises:
            RuntimeError: if a random manager is already registered
                for the name and ``exist_ok=False``.

        Example usage:

        ```pycon

        >>> from coola.random import BaseRandomManager, RandomManager
        >>> class OtherRandomManager(BaseRandomManager):
        ...     def get_rng_state(self) -> dict:
        ...         return {}
        ...     def manual_seed(self, seed: int) -> None:
        ...         pass
        ...     def set_rng_state(self, state: dict) -> dict:
        ...         pass
        ...
        >>> RandomManager.add_manager("other", OtherRandomManager())
        >>> # To overwrite an existing random manager
        >>> RandomManager.add_manager("other", OtherRandomManager(), exist_ok=True)

        ```
        """
        if name in cls.registry and not exist_ok:
            msg = (
                f"A random manager ({cls.registry[name]}) is already registered for the name "
                f"'{name}'. Please use `exist_ok=True` if you want to overwrite the manager for "
                "this name"
            )
            raise RuntimeError(msg)
        cls.registry[name] = manager

    @classmethod
    def has_manager(cls, name: str) -> bool:
        r"""Indicate if a random manager is registered for the given
        name.

        Args:
            name: The name to check.

        Returns:
            ``True`` if a random manager is registered,
                otherwise ``False``.

        Example usage:

        ```pycon

        >>> from coola.random import RandomManager
        >>> RandomManager.has_manager("random")
        True
        >>> RandomManager.has_manager("missing")
        False

        ```
        """
        return name in cls.registry


@contextmanager
def random_seed(seed: int) -> Generator[None]:
    r"""Implement a context manager to manage the random seed and random
    number generator (RNG) state.

    The context manager sets the specified random seed and
    restores the original RNG state afterward.

    Args:
        seed: The random number generator seed to use while using
            this context manager.

    Example usage:

    ```pycon

    >>> import numpy
    >>> from coola.random import random_seed
    >>> with random_seed(42):
    ...     print(numpy.random.randn(2, 4))
    ...
    [[...]]
    >>> with random_seed(42):
    ...     print(numpy.random.randn(2, 4))
    ...
    [[...]]

    ```
    """
    manager = RandomManager()
    state = manager.get_rng_state()
    try:
        manager.manual_seed(seed)
        yield
    finally:
        manager.set_rng_state(state)


def register_random_managers() -> None:
    r"""Register randomness managers to ``RandomManager``.

    Example usage:

    ```pycon

    >>> from coola.random import RandomManager
    >>> from coola.random.default import register_random_managers
    >>> register_random_managers()
    >>> manager = RandomManager()
    >>> manager
    RandomManager(
      (random): RandomRandomManager()
      ...
    )

    ```
    """
    # Local import to avoid cyclic dependency
    from coola.random.utils import get_random_managers

    for name, manager in get_random_managers().items():
        if not RandomManager.has_manager(name):  # pragma: no cover
            RandomManager.add_manager(name, manager)
