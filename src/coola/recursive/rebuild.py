r"""Contain functions to rebuild a container with its original type after
its items were transformed.

The transformers rebuild a container with ``type(data)(items)`` so the
subclasses keep their type. This works for the builtin containers and
the subclasses that do not change ``__init__``. The types with another
constructor need a special case here (``defaultdict``, ``deque``, named
tuples) or a transformer registered for the type. A
``collections.ChainMap`` is rebuilt as a ``ChainMap`` with a single map:
its parents are not part of the structure.
"""

from __future__ import annotations

__all__ = ["rebuild_mapping", "rebuild_sequence", "rebuild_set"]

from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any

from coola.utils.introspection import get_qualified_class_name, get_type_constructor

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


def rebuild_mapping(data: Any, items: Mapping[Any, Any] | Iterable[tuple[Any, Any]]) -> Any:
    r"""Rebuild a mapping of the same type as ``data`` from ``items``.

    ``collections.defaultdict`` is rebuilt with the ``default_factory``
    of ``data``, because its constructor does not accept a mapping as
    the first argument.

    Args:
        data: The original mapping, used for its type.
        items: The new items: a mapping or an iterable of
            ``(key, value)`` pairs.

    Returns:
        A mapping of the same type as ``data``.

    Raises:
        TypeError: If the type of ``data`` cannot be built from
            ``items``. Register a transformer for this type.

    Example:
        ```pycon
        >>> from collections import defaultdict
        >>> from coola.recursive.rebuild import rebuild_mapping
        >>> rebuild_mapping(defaultdict(int, {"a": 1}), {"a": 2})
        defaultdict(<class 'int'>, {'a': 2})

        ```
    """
    constructor = get_type_constructor(data)
    try:
        if isinstance(data, defaultdict):
            return constructor(data.default_factory, items)
        return constructor(items)
    except TypeError as exc:
        raise _rebuild_error(data, exc) from exc


def rebuild_sequence(data: Any, items: Iterable[Any]) -> Any:
    r"""Rebuild a sequence of the same type as ``data`` from ``items``.

    A named tuple is rebuilt from its fields, and a ``collections.deque``
    keeps its ``maxlen``.

    Args:
        data: The original sequence, used for its type.
        items: The new items.

    Returns:
        A sequence of the same type as ``data``.

    Raises:
        TypeError: If the type of ``data`` cannot be built from
            ``items``. Register a transformer for this type.

    Example:
        ```pycon
        >>> from collections import deque
        >>> from coola.recursive.rebuild import rebuild_sequence
        >>> rebuild_sequence(deque([1, 2], maxlen=3), [3, 4])
        deque([3, 4], maxlen=3)

        ```
    """
    constructor = get_type_constructor(data)
    try:
        if isinstance(data, tuple) and hasattr(data, "_fields"):  # namedtuple special case
            return constructor(*items)
        if isinstance(data, deque):
            return constructor(items, maxlen=data.maxlen)
        return constructor(items)
    except TypeError as exc:
        raise _rebuild_error(data, exc) from exc


def rebuild_set(data: Any, items: Iterable[Any]) -> Any:
    r"""Rebuild a set of the same type as ``data`` from ``items``.

    Args:
        data: The original set, used for its type.
        items: The new items. They must be hashable.

    Returns:
        A set of the same type as ``data``. The duplicate items are
        merged.

    Raises:
        TypeError: If an item is not hashable, or if the type of
            ``data`` cannot be built from ``items``.

    Example:
        ```pycon
        >>> from coola.recursive.rebuild import rebuild_set
        >>> rebuild_set(frozenset({1, 2}), [3, 3, 4])
        frozenset({3, 4})

        ```
    """
    try:
        return get_type_constructor(data)(items)
    except TypeError as exc:
        raise _rebuild_error(data, exc) from exc


def _rebuild_error(data: Any, exc: TypeError) -> TypeError:
    r"""Create the error raised when a container cannot be rebuilt.

    Args:
        data: The container.
        exc: The error raised when rebuilding it.

    Returns:
        The error to raise.
    """
    name = get_qualified_class_name(type(data))
    msg = (
        f"Cannot rebuild an object of type {name} from its transformed items. "
        f"If its constructor differs from its base type, register a transformer for {name} "
        f"with `register_transformers`. Original error: {exc}"
    )
    return TypeError(msg)
