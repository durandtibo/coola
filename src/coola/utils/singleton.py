r"""Define a thread-safe lazy singleton helper.

Several modules expose a ``get_default_registry()`` function that builds
and caches a registry on first use. Building it eagerly at import time is
not an option here because some registries are constructed via imports
that are deliberately deferred to avoid circular imports between
``coola.equality`` and ``coola.registry``. ``LazySingleton`` keeps the
construction lazy while making the first-call race thread-safe via
double-checked locking.
"""

from __future__ import annotations

__all__ = ["LazySingleton", "make_default_registry_singleton"]

import threading
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")
R = TypeVar("R")


class LazySingleton(Generic[T]):
    r"""Thread-safe holder that builds a value lazily on first access
    and caches it.

    Args:
        factory: Callable used to build the value on first access. It
            is called at most once, even if multiple threads call
            ``get`` concurrently before the value exists.

    Example:
        ```pycon
        >>> from coola.utils.singleton import LazySingleton
        >>> counter = {"calls": 0}
        >>> def build() -> int:
        ...     counter["calls"] += 1
        ...     return 42
        ...
        >>> singleton = LazySingleton(build)
        >>> singleton.get()
        42
        >>> singleton.get()
        42
        >>> counter["calls"]
        1

        ```
    """

    def __init__(self, factory: Callable[[], T]) -> None:
        self._factory = factory
        self._instance: T | None = None
        self._lock = threading.Lock()

    def get(self) -> T:
        r"""Return the cached value, building it on the first call.

        Returns:
            The singleton value.
        """
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    self._instance = self._factory()
        return self._instance


def make_default_registry_singleton(
    registry_cls: Callable[[], R],
    register_defaults: Callable[[R], None],
) -> LazySingleton[R]:
    r"""Create a lazy singleton that builds and populates a default
    registry on first access.

    This removes the near-identical ``_build_default_registry``
    functions previously copied in each package, and guarantees the same
    thread-safe construction everywhere.

    Args:
        registry_cls: Callable (usually the registry class) that
            returns a new, empty registry.
        register_defaults: Callable that populates the registry with
            the default entries.

    Returns:
        A ``LazySingleton`` whose ``get`` method returns the populated
            registry.

    Example:
        ```pycon
        >>> from coola.utils.singleton import make_default_registry_singleton
        >>> singleton = make_default_registry_singleton(dict, lambda r: r.update(a=1))
        >>> singleton.get()
        {'a': 1}

        ```
    """

    def build() -> R:
        registry = registry_cls()
        register_defaults(registry)
        return registry

    return LazySingleton(build)
