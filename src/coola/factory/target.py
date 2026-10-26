r"""Implement a function to know what a target builds."""

from __future__ import annotations

__all__ = ["builds"]

import inspect
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints


def builds(target: Any, cls: type | tuple[type, ...]) -> bool | None:
    r"""Indicate if calling ``target`` builds an instance of ``cls``,
    from the types only.

    The target is not called. A class builds itself. A function builds
    the class of its return type hint. A union (``A | B``) builds one
    of its members, and a generic (``list[int]``) builds its origin
    (``list``).

    Args:
        target: The target, a class, a function or any object.
        cls: The expected class, or a tuple of classes (as accepted
            by ``issubclass``).

    Returns:
        ``True`` if the classes that ``target`` builds include a
            subclass of ``cls``, ``False`` if none of them is, and
            ``None`` if it cannot be known: a built-in function, a
            function without return type hint, a hint that cannot be
            resolved or that is not a class (``Any``), or a union with
            such a member.

    Example:
        ```pycon
        >>> from collections import Counter
        >>> from coola.factory.target import builds
        >>> builds(Counter, dict)
        True
        >>> builds(Counter, list)
        False
        >>> builds(print, dict) is None
        True

        ```
    """
    if inspect.isfunction(target):
        try:
            target = get_type_hints(target).get("return")
        except (NameError, SyntaxError, TypeError):
            # The return type hint cannot be resolved (e.g. a forward reference
            # to a name that is not defined): it cannot be checked.
            return None
    if target is None:
        return None
    # Union[T1, T2] or T1 | T2
    members = get_args(target) if get_origin(target) in (Union, UnionType) else (target,)
    # A generic alias such as ``list[int]`` is not a class: check its origin (``list``).
    classes = tuple(get_origin(member) or member for member in members)
    if not all(isinstance(c, type) and c is not Any for c in classes):  # Any is a class in 3.11+
        return None
    return any(issubclass(c, cls) for c in classes)
