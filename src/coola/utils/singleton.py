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

__all__ = [
    "STRICT_PLUGINS_ENV_VAR",
    "LazySingleton",
    "load_registry_plugins",
    "make_default_registry_singleton",
]

import logging
import os
import threading
import warnings
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable
    from importlib.metadata import EntryPoint

logger: logging.Logger = logging.getLogger(__name__)

#: Environment variable that makes a failing plugin raise instead of being
#: skipped with a warning. It is enabled by any value other than ``""``,
#: ``"0"`` and ``"false"`` (case-insensitive), and is useful in CI.
STRICT_PLUGINS_ENV_VAR = "COOLA_STRICT_PLUGINS"

T = TypeVar("T")
R = TypeVar("R")


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
        self._instance: T | None = None
        self._lock = threading.Lock()
        self._building_thread: int | None = None

    def get(self) -> T:
        r"""Return the cached value, building it on the first call.

        Returns:
            The singleton value.
        """
        if self._instance is None:
            if self._building_thread == threading.get_ident():
                msg = (
                    "The value is requested while it is being built, which would wait "
                    "forever. Use the object that is being built instead of requesting "
                    "the singleton again (for a plugin of a default registry, use the "
                    "registry that the plugin receives)."
                )
                raise RuntimeError(msg)
            with self._lock:
                if self._instance is None:
                    self._building_thread = threading.get_ident()
                    try:
                        self._instance = self._factory()
                    finally:
                        self._building_thread = None
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
    plugin cannot break coola. The failure is also logged with its
    traceback. If the ``COOLA_STRICT_PLUGINS`` environment variable is
    set (e.g. in CI), every plugin is still attempted and all failures
    are then raised together as an ``ExceptionGroup``, each annotated
    with the plugin name and group.

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
    errors: list[Exception] = []
    for entry_point in entry_points(group=group):
        _load_plugin(entry_point, registry, group, errors)
    if errors:
        msg = f"Failed to load {len(errors)} plugin(s) of group {group!r}"
        raise ExceptionGroup(msg, errors)


def _strict_plugins() -> bool:
    r"""Indicate whether plugin failures must raise instead of warn."""
    return os.environ.get(STRICT_PLUGINS_ENV_VAR, "").strip().lower() not in {"", "0", "false"}


def _load_plugin(
    entry_point: EntryPoint, registry: object, group: str, errors: list[Exception]
) -> None:
    r"""Load one plugin and apply it to the registry.

    A failure is logged with its traceback and reported as a
    ``RuntimeWarning``, or, in strict mode, annotated with a note and
    appended to ``errors`` so the caller can raise them together.
    """
    try:
        entry_point.load()(registry)
    except Exception as exc:
        if _strict_plugins():
            exc.add_note(f"while loading plugin {entry_point.name!r} of group {group!r}")
            errors.append(exc)
            return
        logger.exception("Failed to load plugin %r of group %r", entry_point.name, group)
        warnings.warn(
            f"Skipping plugin {entry_point.name!r} of group {group!r}: {exc!r}",
            RuntimeWarning,
            stacklevel=3,
        )
