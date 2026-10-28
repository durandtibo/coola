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

__all__ = ["LazySingleton"]

import threading
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from coola.utils.fork import track_fork

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")

_UNSET: Any = object()


class LazySingleton(Generic[T]):
    r"""Thread-safe holder that builds a value lazily on first access
    and caches it.

    Args:
        factory: Callable used to build the value on first access. It
            is called at most once, even if multiple threads call
            ``get`` concurrently before the value exists.

    Raises:
        RuntimeError: If ``get`` is called by the factory itself (on
            the same thread) while the value is being built. The value
            does not exist yet, so waiting for it would block forever.
            For a default registry, this happens when a plugin calls a
            function that uses the registry that is being built; the
            plugin should use the registry it receives instead.

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
        self._instance: T = _UNSET
        self._lock = threading.Lock()
        self._building_thread: int | None = None
        track_fork(self)

    def _reset_after_fork(self) -> None:
        r"""Recreate the lock in a forked child process.

        The lock may have been held by a thread that does not exist in
        the child process (the thread that was building the value),
        which would block the child forever. If the value was not built
        yet, the child builds it.
        """
        self._lock = threading.Lock()
        self._building_thread = None

    def get(self) -> T:
        r"""Return the cached value, building it on the first call.

        Returns:
            The singleton value.

        Raises:
            RuntimeError: If the value is requested again by the thread
                that is building it (the factory requests the
                singleton it builds), which would otherwise wait
                forever.
        """
        if self._instance is _UNSET:
            if self._building_thread == threading.get_ident():
                msg = (
                    "The value is requested while it is being built, which would wait "
                    "forever. Use the object that is being built instead of requesting "
                    "the singleton again (for a plugin of a default registry, use the "
                    "registry that the plugin receives)."
                )
                raise RuntimeError(msg)
            with self._lock:
                if self._instance is _UNSET:
                    self._building_thread = threading.get_ident()
                    try:
                        self._instance = self._factory()
                    finally:
                        self._building_thread = None
        return self._instance

    def is_built(self) -> bool:
        r"""Indicate if the value has been built.

        Returns:
            ``True`` if the value has been built, otherwise ``False``.
        """
        return self._instance is not _UNSET

    def _check_not_building(self, action: str) -> None:
        r"""Raise an error if the current thread is building the value,
        because ``action`` would wait for a lock that this thread holds."""
        if self._building_thread == threading.get_ident():
            msg = (
                f"The singleton is {action} while its value is being built, which would wait "
                f"forever. Do not {action} the singleton from its own factory."
            )
            raise RuntimeError(msg)

    def reset(self) -> None:
        r"""Forget the cached value, so the next ``get`` call builds a
        new one.

        Raises:
            RuntimeError: If ``reset`` is called by the thread that is
                building the value (the factory resets the singleton it
                builds), which would otherwise wait forever.
        """
        self._check_not_building("reset")
        with self._lock:
            self._instance = _UNSET

    def set(self, value: T) -> None:
        r"""Replace the cached value.

        A reference to the previous value is not changed. If the value
        was not built yet, the factory is not needed anymore.

        Args:
            value: The new value.

        Raises:
            RuntimeError: If ``set`` is called by the thread that is
                building the value (the factory sets the singleton it
                builds), which would otherwise wait forever.

        Example:
            ```pycon
            >>> from coola.utils.singleton import LazySingleton
            >>> singleton = LazySingleton(lambda: 1)
            >>> singleton.set(2)
            >>> singleton.get()
            2

            ```
        """
        self._check_not_building("set")
        with self._lock:
            self._instance = value
