r"""Define the default registries of the packages built on
``coola.registry``."""

from __future__ import annotations

__all__ = [
    "make_default_registry_singleton",
    "reset_default_registries",
    "temporary_default_registries",
]

import weakref
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, TypeVar

from coola.registry.plugins import load_registry_plugins
from coola.utils.singleton import LazySingleton

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

R = TypeVar("R")

_UNSET: Any = object()

# The singletons created by ``make_default_registry_singleton``.
_DEFAULT_REGISTRY_SINGLETONS: weakref.WeakSet[LazySingleton[Any]] = weakref.WeakSet()


def make_default_registry_singleton(
    registry_cls: Callable[[], R],
    register_defaults: Callable[[R], None],
    *,
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
        >>> from coola.registry.default import make_default_registry_singleton
        >>> singleton = make_default_registry_singleton(dict, lambda r: r.update(a=1))
        >>> singleton.get()
        {'a': 1}

        ```
    """

    def build() -> R:
        r"""Build the default registry.

        The registry is created, filled with the defaults, then with the
        plugins.

        Returns:
            The populated registry.
        """
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
        >>> from coola.registry.default import reset_default_registries
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

    Yields:
        ``None``. The default registries are restored when the block
            exits.

    Example:
        ```pycon
        >>> from coola.equality.tester import DefaultEqualityTester, get_default_registry
        >>> from coola.registry.default import temporary_default_registries
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
        (singleton, singleton.get() if singleton.is_built() else _UNSET)
        for singleton in _DEFAULT_REGISTRY_SINGLETONS
    ]
    for singleton, value in previous:
        if value is not _UNSET:
            singleton.set(value.copy())
    try:
        yield
    finally:
        for singleton, value in previous:
            if value is _UNSET:
                singleton.reset()
            else:
                singleton.set(value)
