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

__all__ = ["LazySingleton", "load_registry_plugins", "make_default_registry_singleton"]

import threading
import warnings
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable
    from importlib.metadata import EntryPoint

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
    plugin_group: str | None = None,
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
        plugin_group: Optional entry-point group. If set, every entry
            point in this group is loaded after the defaults and called
            with the registry, so third-party packages can register
            their types without touching coola (see
            ``load_registry_plugins``).

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
        if plugin_group is not None:
            load_registry_plugins(registry, plugin_group)
        return registry

    return LazySingleton(build)


def load_registry_plugins(registry: object, group: str) -> None:
    r"""Populate a registry from third-party entry points.

    Each entry point in ``group`` must resolve to a callable that takes
    the registry and registers its types on it. A plugin that fails to
    load or run emits a ``RuntimeWarning`` and is skipped, so one broken
    plugin cannot break coola.

    Args:
        registry: The registry passed to each plugin.
        group: The entry-point group to load, e.g.
            ``"coola.equality.testers"``.

    Example:
        ```pycon
        >>> from coola.utils.singleton import load_registry_plugins
        >>> load_registry_plugins({}, "coola.nonexistent.group")

        ```
    """
    for entry_point in entry_points(group=group):
        _load_plugin(entry_point, registry, group)


def _load_plugin(entry_point: EntryPoint, registry: object, group: str) -> None:
    r"""Load one plugin and apply it to the registry, warning on
    failure."""
    try:
        entry_point.load()(registry)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"Skipping plugin {entry_point.name!r} of group {group!r}: {exc!r}",
            RuntimeWarning,
            stacklevel=3,
        )
