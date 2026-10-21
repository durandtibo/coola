r"""Contain functions to rebuild a container with its original type
after its items were transformed."""

from __future__ import annotations

__all__ = ["rebuild_mapping", "rebuild_sequence"]

from collections import defaultdict, deque
from typing import Any

from coola.utils.introspection import get_type_constructor


def rebuild_mapping(data: Any, items: dict[Any, Any]) -> Any:
    r"""Rebuild a mapping of the same type as ``data`` from ``items``.

    ``collections.defaultdict`` is rebuilt with the ``default_factory``
    of ``data``, because its constructor does not accept a mapping as
    the first argument.

    Args:
        data: The original mapping, used for its type.
        items: The new items.

    Returns:
        A mapping of the same type as ``data``.

    Example:
        ```pycon
        >>> from collections import defaultdict
        >>> from coola.recursive.rebuild import rebuild_mapping
        >>> rebuild_mapping(defaultdict(int, {"a": 1}), {"a": 2})
        defaultdict(<class 'int'>, {'a': 2})

        ```
    """
    if isinstance(data, defaultdict):
        return get_type_constructor(data)(data.default_factory, items)
    return get_type_constructor(data)(items)


def rebuild_sequence(data: Any, items: list[Any]) -> Any:
    r"""Rebuild a sequence of the same type as ``data`` from ``items``.

    A named tuple is rebuilt from its fields, and a ``collections.deque``
    keeps its ``maxlen``.

    Args:
        data: The original sequence, used for its type.
        items: The new items.

    Returns:
        A sequence of the same type as ``data``.

    Example:
        ```pycon
        >>> from collections import deque
        >>> from coola.recursive.rebuild import rebuild_sequence
        >>> rebuild_sequence(deque([1, 2], maxlen=3), [3, 4])
        deque([3, 4], maxlen=3)

        ```
    """
    constructor = get_type_constructor(data)
    if isinstance(data, tuple) and hasattr(data, "_fields"):  # namedtuple special case
        return constructor(*items)
    if isinstance(data, deque):
        return constructor(items, maxlen=data.maxlen)
    return constructor(items)
