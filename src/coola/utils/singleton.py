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
    "DISABLE_PLUGINS_ENV_VAR",
    "STRICT_PLUGINS_ENV_VAR",
    "LazySingleton",
    "LoadedPlugin",
    "get_loaded_plugins",
    "load_registry_plugins",
    "make_default_registry_singleton",
    "reset_default_registries",
    "temporary_default_registries",
]

import logging
import os
import threading
import warnings
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from coola.utils.fork import track_fork

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from importlib.metadata import EntryPoint

logger: logging.Logger = logging.getLogger(__name__)

#: Environment variable that makes a failing plugin raise instead of being
#: skipped with a warning. It is enabled by any value other than ``""``,
#: ``"0"`` and ``"false"`` (case-insensitive), and is useful in CI.
STRICT_PLUGINS_ENV_VAR = "COOLA_STRICT_PLUGINS"

#: Environment variable that disables the entry-point plugins: none of them
#: is loaded. It is enabled by any value other than ``""``, ``"0"`` and
#: ``"false"`` (case-insensitive), and is useful to get a reproducible run,
#: or to find out which plugin causes a conflict.
DISABLE_PLUGINS_ENV_VAR = "COOLA_DISABLE_PLUGINS"

T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class LoadedPlugin:
    r"""A plugin that was loaded and applied to a registry.

    Attributes:
        group: The entry-point group of the plugin, for example
            ``"coola.equality.testers"``.
        name: The name of the entry point.
        value: The target of the entry point, for example
            ``"mypackage.plugin:register"``.
    """

    group: str
    name: str
    value: str


# The plugins that were applied, by (group, name). A dict, so the plugins that are applied again
# (a default registry that is rebuilt) are listed once.
_LOADED_PLUGINS: dict[tuple[str, str], LoadedPlugin] = {}


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

    def reset(self) -> None:
        r"""Forget the cached value, so the next ``get`` call builds a
        new one.

        Raises:
            RuntimeError: If ``reset`` is called by the thread that is
                building the value (the factory resets the singleton it
                builds), which would otherwise wait forever.
        """
        if self._building_thread == threading.get_ident():
            msg = (
                "The singleton is reset while its value is being built, which would wait "
                "forever. Do not reset the singleton from its own factory."
            )
            raise RuntimeError(msg)
        with self._lock:
            self._instance = _UNSET

    def _replace_with_copy(self) -> T:
        r"""Replace the cached value by a copy of it and return the
        previous value.

        If the value is not built yet, nothing is replaced and the
        returned value is ``_UNSET``.
        """
        with self._lock:
            previous = self._instance
            if previous is not _UNSET:
                self._instance = cast("Any", previous).copy()
            return previous

    def _restore(self, value: T) -> None:
        r"""Set the cached value (``_UNSET`` forgets it)."""
        with self._lock:
            self._instance = value


# The singletons created by ``make_default_registry_singleton``.
_DEFAULT_REGISTRY_SINGLETONS: weakref.WeakSet[LazySingleton[Any]] = weakref.WeakSet()


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

    singleton = LazySingleton(build)
    _DEFAULT_REGISTRY_SINGLETONS.add(singleton)
    return singleton


def reset_default_registries() -> None:
    r"""Forget the default registries, so the next use of each of them
    builds a new one.

    Every default registry (equality testers, hashers, summarizers,
    ...) is rebuilt with its default entries and its plugins, and the
    registrations that were made with the ``register_*`` functions, and
    the calls to ``clear`` or ``unregister``, are dropped. A registry
    that was obtained before the reset is not changed.

    Example:
        ```pycon
        >>> from coola.equality.tester import get_default_registry
        >>> from coola.utils.singleton import reset_default_registries
        >>> get_default_registry().clear()
        >>> reset_default_registries()
        >>> get_default_registry().has_equality_tester(int)
        True

        ```
    """
    for singleton in list(_DEFAULT_REGISTRY_SINGLETONS):
        singleton.reset()


@contextmanager
def temporary_default_registries() -> Generator[None]:
    r"""Restore the default registries when the context exits.

    Inside the context, the default registries that are already built
    are replaced by a copy (``copy``), so a registration, an
    ``unregister`` or a ``clear`` does not change the original. The
    default registries that are not built yet are built inside the
    context and forgotten when it exits. It is meant for tests that
    register a type, so they do not leak the registration to the other
    tests of the process.

    The default registries are process-wide: do not enter the context
    in several threads at the same time.

    Example:
        ```pycon
        >>> from coola.equality.tester import DefaultEqualityTester, get_default_registry
        >>> from coola.utils.singleton import temporary_default_registries
        >>> class Point:
        ...     pass
        ...
        >>> with temporary_default_registries():
        ...     get_default_registry().register(Point, DefaultEqualityTester())
        ...     get_default_registry().has_equality_tester(Point)
        ...
        True
        >>> get_default_registry().has_equality_tester(Point)
        False

        ```
    """
    previous = [
        (singleton, singleton._replace_with_copy()) for singleton in _DEFAULT_REGISTRY_SINGLETONS
    ]
    try:
        yield
    finally:
        for singleton, value in previous:
            singleton._restore(value)


def get_loaded_plugins() -> list[LoadedPlugin]:
    r"""Get the entry-point plugins that were loaded in this process.

    A plugin is listed after it was applied to a default registry. The
    default registries load their plugins the first time they are
    built, so a registry that was not used yet has not loaded its
    plugins. It makes it possible to find out which installed package
    changed the behaviour of ``objects_are_equal``, ``hash_object`` or
    ``summarize``, without reading ``importlib.metadata``.

    Returns:
        The loaded plugins, in the order in which they were loaded.
        The plugins that failed to load, and all the plugins when
        ``COOLA_DISABLE_PLUGINS`` is set, are not listed.

    Example:
        ```pycon
        >>> from coola.utils.singleton import get_loaded_plugins
        >>> plugins = get_loaded_plugins()
        >>> isinstance(plugins, list)
        True

        ```
    """
    return list(_LOADED_PLUGINS.values())


def load_registry_plugins(registry: object, group: str) -> None:
    r"""Populate a registry from third-party entry points.

    Each entry point in ``group`` must resolve to a callable that takes
    the registry and registers its types on it. A plugin that fails to
    load or run emits a ``RuntimeWarning`` and is skipped, so one broken
    plugin cannot break coola. The failure is also logged with its
    traceback. If the ``COOLA_STRICT_PLUGINS`` environment variable is
    set (e.g. in CI), every plugin is still attempted and all failures
    are then raised together as an ``ExceptionGroup``, each annotated
    with the plugin name and group. If the ``COOLA_DISABLE_PLUGINS``
    environment variable is set, no plugin is loaded. The plugins that
    were loaded are listed by ``get_loaded_plugins``.

    Args:
        registry: The registry passed to each plugin.
        group: The entry-point group to load, e.g.
            ``"coola.equality.testers"``.

    Raises:
        ExceptionGroup: If ``COOLA_STRICT_PLUGINS`` is set and at least
            one plugin failed to load or run.

    Example:
        ```pycon
        >>> from coola.utils.singleton import load_registry_plugins
        >>> load_registry_plugins({}, "coola.nonexistent.group")

        ```
    """
    if _plugins_disabled():
        return
    errors: list[Exception] = []
    for entry_point in entry_points(group=group):
        _load_plugin(entry_point, registry, group, errors)
    if errors:
        msg = f"Failed to load {len(errors)} plugin(s) of group {group!r}"
        raise ExceptionGroup(msg, errors)


def _env_flag(name: str) -> bool:
    r"""Indicate whether an environment variable is enabled (set to a
    value other than ``""``, ``"0"`` and ``"false"``)."""
    return os.environ.get(name, "").strip().lower() not in {"", "0", "false"}


def _strict_plugins() -> bool:
    r"""Indicate whether plugin failures must raise instead of warn."""
    return _env_flag(STRICT_PLUGINS_ENV_VAR)


def _plugins_disabled() -> bool:
    r"""Indicate whether the plugins must not be loaded."""
    return _env_flag(DISABLE_PLUGINS_ENV_VAR)


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
        _LOADED_PLUGINS[group, entry_point.name] = LoadedPlugin(
            group=group, name=entry_point.name, value=entry_point.value
        )
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
