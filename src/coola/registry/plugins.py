r"""Load the entry-point plugins that populate a registry."""

from __future__ import annotations

__all__ = [
    "DISABLE_PLUGINS_ENV_VAR",
    "STRICT_PLUGINS_ENV_VAR",
    "LoadedPlugin",
    "get_loaded_plugins",
    "load_registry_plugins",
]

import logging
import os
import warnings
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
        >>> from coola.registry.plugins import get_loaded_plugins
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
        >>> from coola.registry.plugins import load_registry_plugins
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
