r"""Implement a function to check if a configuration matches a given
class.

This module provides functionality to indicate if the input
configuration is a configuration for a given class.
"""

from __future__ import annotations

__all__ = ["is_object_config"]

import inspect
from types import UnionType
from typing import TYPE_CHECKING, Any, Union, get_args, get_origin, get_type_hints

from coola.factory.constants import OBJECT_TARGET
from coola.factory.instantiation import import_object

if TYPE_CHECKING:
    from collections.abc import Sequence


def is_object_config(
    config: dict[str, Any], cls: type, *, allowed_prefixes: Sequence[str] | None = None
) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    given class.

    This function only checks if the value of the key ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class. A function whose return type hint cannot be resolved
    (e.g. a forward reference to an undefined name) is not a
    configuration for ``cls``.

    Args:
        config: The configuration to check.
        cls: The object class.
        allowed_prefixes: An optional allow-list of module prefixes
            that ``_target_`` must match (see ``import_object``).
            If ``None`` (default), any ``_target_`` is allowed.

    Returns:
        ``True`` if the input configuration is a configuration
            for the given class.

    Raises:
        ImportError: if ``_target_`` does not match
            ``allowed_prefixes``, or if it cannot be imported.

    Security:
        The ``_target_`` is **imported** to check its class, and
        importing a module runs its top-level code. Do not call this
        function with a ``_target_`` that comes from an untrusted
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
    target = config.get(OBJECT_TARGET)
    if target is None:
        return False
    target = import_object(target, allowed_prefixes=allowed_prefixes)
    if inspect.isfunction(target):
        try:
            target = get_type_hints(target).get("return")
        except (NameError, TypeError):
            # The return type hint cannot be resolved (e.g. a forward reference
            # to a name that is not defined): it cannot be checked.
            return False
    if target is None:
        return False
    origin = get_origin(target)
    # Union[T1, T2] or T1 | T2
    targets = get_args(target) if origin in (Union, UnionType) else (target,)
    # A generic alias such as ``list[int]`` is not a class: check its origin (``list``).
    targets = tuple(get_origin(t) or t for t in targets)
    return any(inspect.isclass(t) and issubclass(t, cls) for t in targets)
