r"""Implement a function to check if a configuration matches a given
class.

This module provides functionality to indicate if the input
configuration is a configuration for a given class.
"""

from __future__ import annotations

__all__ = ["is_object_config"]

from typing import TYPE_CHECKING, Any

from coola.factory.constants import OBJECT_TARGET
from coola.factory.instantiation import import_object
from coola.factory.target import builds
from coola.utils.imports import is_module_imported

if TYPE_CHECKING:
    from collections.abc import Sequence


def is_object_config(
    config: dict[str, Any],
    cls: type,
    *,
    allowed_prefixes: Sequence[str] | None = None,
    import_target: bool = False,
) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    given class.

    This function only checks if the value of the key ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class. A function whose return type hint cannot be resolved
    (e.g. a forward reference to an undefined name, or a string
    annotation that is not valid Python) is not a configuration for
    ``cls``. A ``config`` that is not a ``dict``, or whose ``_target_``
    is not a string, is not a configuration for ``cls``.

    The module of ``_target_`` is not imported to answer: if it is not
    imported yet, the configuration is not a configuration for ``cls``
    (importing a module runs its top-level code). Use
    ``allowed_prefixes`` to restrict the modules that can be imported,
    or ``import_target=True`` to import the module of any ``_target_``.

    Args:
        config: The configuration to check.
        cls: The object class.
        allowed_prefixes: An optional allow-list of module prefixes
            that ``_target_`` must match (see ``import_object``).
            If ``None`` (default), any ``_target_`` is allowed, but its
            module is imported only if ``import_target`` is ``True``.
            If it is set, the module of a ``_target_`` that matches is
            imported when needed.
        import_target: If ``True``, the module of ``_target_`` is
            imported if it is not imported yet, even without
            ``allowed_prefixes``. If ``False`` (default), a
            ``_target_`` whose module is not imported yet and that is
            not restricted by ``allowed_prefixes`` is not a
            configuration for ``cls``.

    Returns:
        ``True`` if the input configuration is a configuration
            for the given class.

    Raises:
        ImportError: if ``_target_`` does not match
            ``allowed_prefixes``, or if it cannot be imported.

    Security:
        By default, the module of the ``_target_`` is **not** imported,
        so a configuration from an untrusted source cannot run the
        top-level code of an arbitrary module. With ``allowed_prefixes``
        or ``import_target=True``, the module is **imported** to check
        the class, and importing a module runs its top-level code: do
        not use them with a ``_target_`` that comes from an untrusted
        source, unless ``allowed_prefixes`` restricts the modules that
        can be imported.

    Example:
        ```pycon
        >>> from coola.factory import is_object_config
        >>> from collections import Counter
        >>> is_object_config({"_target_": "collections.Counter", "iterable": [1, 2, 1, 3]}, Counter)
        True

        ```
    """
    if not isinstance(config, dict):
        return False
    target = config.get(OBJECT_TARGET)
    if not isinstance(target, str):
        return False
    if not (import_target or allowed_prefixes is not None or is_module_imported(target)):
        return False
    return builds(import_object(target, allowed_prefixes=allowed_prefixes), cls) is True
