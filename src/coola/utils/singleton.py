r"""Implement a thread-safe lazy singleton helper.

This module factors out the ``get_default_registry()`` singleton pattern
that is duplicated across several ``coola`` subpackages
(``equality.tester``, ``hashing``, ``recursive``, ``random``,
``iterator.bfs``, ``iterator.dfs``, ``summary``).
"""

from __future__ import annotations

__all__ = ["LazySingleton"]

import threading
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")


class LazySingleton(Generic[T]):
    r"""Implement a thread-safe holder that lazily creates and caches a
    single instance.

    The wrapped ``factory`` is called at most once: the first call to
    ``get`` creates and caches the instance, and every subsequent call
    returns the cached instance. Double-checked locking is used so
    that two threads calling ``get`` concurrently before the instance
    exists cannot each build and store a different instance.

    Args:
        factory: A callable with no arguments that creates the
            singleton instance.

    Example:
        ```pycon
        >>> from coola.utils.singleton import LazySingleton
        >>> counter = {"n": 0}
        >>> def factory():
        ...     counter["n"] += 1
        ...     return object()
        ...
        >>> singleton = LazySingleton(factory)
        >>> x = singleton.get()
        >>> y = singleton.get()
        >>> x is y
        True
        >>> counter["n"]
        1

        ```
    """

    def __init__(self, factory: Callable[[], T]) -> None:
        self._factory = factory
        self._instance: T | None = None
        self._lock = threading.Lock()

    def get(self) -> T:
        r"""Return the singleton instance, creating it on first call.

        Returns:
            The cached singleton instance.
        """
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    self._instance = self._factory()
        return self._instance

    def reset(self) -> None:
        r"""Clear the cached instance so the next call to ``get``
        recreates it.

        Mainly useful in tests that need a fresh instance between cases.
        """
        with self._lock:
            self._instance = None
