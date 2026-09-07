r"""Implement a function to check if a configuration matches a given
class.

This module provides functionality to indicate if the input
configuration is a configuration for a given class.
"""

from __future__ import annotations

__all__ = ["is_object_config"]

import inspect
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

from coola.factory.constants import OBJECT_TARGET
from coola.factory.instantiation import import_object


def is_object_config(config: dict[str, Any], cls: type) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    given class.

    This function only checks if the value of the key ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
        config: The configuration to check.
        cls: The object class.

    Returns:
        ``True`` if the input configuration is a configuration
            for the given class.

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
    target = import_object(target)
    if inspect.isfunction(target):
        target = get_type_hints(target).get("return")
    if target is None:
        return False
    origin = get_origin(target)
    # Union[T1, T2] or T1 | T2
    targets = get_args(target) if origin in (Union, UnionType) else (target,)
    return any(inspect.isclass(t) and cls in t.__mro__ for t in targets)
