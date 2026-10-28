r"""Define the public interface to iterate over the leaves of nested
data."""

from __future__ import annotations

__all__ = [
    "bfs_iterate",
    "dfs_items",
    "dfs_iterate",
    "get_default_child_finder_registry",
    "get_default_registry",
    "register_child_finders",
]

import dataclasses
from collections import UserString, deque
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from coola.iterator.dataclass import DataclassChildFinder
from coola.iterator.default import DefaultChildFinder
from coola.iterator.engine import breadth_first, depth_first, depth_first_items
from coola.iterator.iterable import IterableChildFinder
from coola.iterator.mapping import MappingChildFinder
from coola.iterator.registry import ChildFinderRegistry
from coola.registry.default import make_default_registry_singleton

if TYPE_CHECKING:
    from collections.abc import Iterator

    from coola.iterator.base import BaseChildFinder
    from coola.utils.singleton import LazySingleton


def dfs_iterate(data: object, *, registry: ChildFinderRegistry | None = None) -> Iterator[Any]:
    """Iterate over the leaves of nested data in depth-first order.

    The traversal is pre-order, left to right. The child finder of each
    object is chosen from its type with ``registry``. Only the leaves
    are yielded, once per occurrence, never a container (an empty
    container yields nothing). The keys of a mapping are never yielded.
    Iterators and generators are leaves: they are never consumed.

    The function is a generator: the leaves are produced on demand and
    nothing is walked before the first ``next``. It uses an explicit
    stack, so there is no limit on the depth of the data.

    Args:
        data: The nested data to traverse.
        registry: The registry used to resolve the child finders. If
            ``None``, the default registry is used.

    Yields:
        The leaves of ``data`` in depth-first order.

    Raises:
        ValueError: If a container contains itself, when the traversal
            reaches it (the leaves before it are already yielded).

    Example:
        ```pycon
        >>> from coola.iterator import dfs_iterate
        >>> list(dfs_iterate({"a": 1, "b": "abc"}))
        [1, 'abc']
        >>> list(dfs_iterate([1, [2, 3], {"x": 4}]))
        [1, 2, 3, 4]

        ```
    """
    if registry is None:
        registry = get_default_registry()
    return depth_first(data, registry)


def dfs_items(
    data: object, *, registry: ChildFinderRegistry | None = None, empty: bool = False
) -> Iterator[tuple[tuple[Any, ...], Any]]:
    """Iterate over the leaves of nested data with their path, in depth-
    first order.

    It yields the leaves of ``dfs_iterate`` in the same order, each
    with its path: the tuple of the labels of the containers crossed
    to reach it. The label of a child is given by the ``find_items``
    method of its container's child finder: the key of a mapping, the
    index of a sequence, the name of the field of a dataclass, the
    element itself for a set. The labels are raw (a key is not
    converted to ``str``). The root being a leaf gives the empty path.

    The function is a generator and uses an explicit stack, so there
    is no limit on the depth of the data.

    Args:
        data: The nested data to traverse.
        registry: The registry used to resolve the child finders. If
            ``None``, the default registry is used.
        empty: If ``True``, an empty container is also yielded, as a
            leaf (``(path, container)``). By default an empty
            container yields nothing.

    Yields:
        The pairs ``(path, leaf)`` of ``data`` in depth-first order.

    Raises:
        ValueError: If a container contains itself, when the traversal
            reaches it (the leaves before it are already yielded).

    Example:
        ```pycon
        >>> from coola.iterator import dfs_items
        >>> list(dfs_items({"a": 1, "b": [2, {"c": 3}]}))
        [(('a',), 1), (('b', 0), 2), (('b', 1, 'c'), 3)]
        >>> list(dfs_items({"a": {}}))
        []
        >>> list(dfs_items({"a": {}}, empty=True))
        [(('a',), {})]

        ```
    """
    if registry is None:
        registry = get_default_registry()
    return depth_first_items(data, registry, empty=empty)


def bfs_iterate(data: object, *, registry: ChildFinderRegistry | None = None) -> Iterator[Any]:
    """Iterate over the leaves of nested data in breadth-first order.

    The traversal is level order, left to right inside a level. It
    visits the same leaves as ``dfs_iterate``; only the order differs.
    The child finder of each object is chosen from its type with
    ``registry``. Only the leaves are yielded, never a container.

    The function is a generator: the leaves are produced on demand. It
    uses an explicit queue, so there is no limit on the depth of the
    data. The memory is proportional to the width of a level.

    Args:
        data: The nested data to traverse.
        registry: The registry used to resolve the child finders. If
            ``None``, the default registry is used.

    Yields:
        The leaves of ``data`` in breadth-first order.

    Raises:
        ValueError: If a container contains itself, when the traversal
            reaches it (the leaves before it are already yielded).

    Example:
        ```pycon
        >>> from coola.iterator import bfs_iterate
        >>> list(bfs_iterate({"a": 1, "b": "abc"}))
        [1, 'abc']
        >>> list(bfs_iterate([1, [2, [3]], 4]))
        [1, 4, 2, 3]

        ```
    """
    if registry is None:
        registry = get_default_registry()
    return breadth_first(data, registry)


def register_child_finders(
    mapping: Mapping[type, BaseChildFinder[Any]],
    *,
    exist_ok: bool = False,
) -> None:
    """Register custom child finders to the default global registry.

    One registration serves both ``dfs_iterate`` and ``bfs_iterate``.

    Args:
        mapping: A dictionary mapping Python types to their corresponding child finder instances.
        exist_ok: If `True`, existing registrations for types will be overwritten.
            If `False`, an error is raised when a type is already registered.

    Example:
        ```pycon
        >>> from coola.iterator import IterableChildFinder, register_child_finders
        >>> register_child_finders({list: IterableChildFinder()}, exist_ok=True)

        ```
    """
    get_default_registry().register_many(mapping, exist_ok=exist_ok)


def get_default_registry() -> ChildFinderRegistry:
    """Get or create the default global registry for child finders.

    This function returns a singleton instance of the `ChildFinderRegistry`, which is
    pre-configured with child finders for the common Python types: the sequences, the sets,
    the mappings and the dataclasses are entered, the scalars and the other objects are leaves.

    Returns:
        An `ChildFinderRegistry` instance with child finders registered for common Python types.

    Notes:
        The singleton pattern means any changes to the returned registry affect all future
        calls to this function. If an isolated registry is needed, create a new `ChildFinderRegistry`
        instance directly.

    Example:
        ```pycon
        >>> from coola.iterator import get_default_registry
        >>> registry = get_default_registry()
        >>> registry.has_child_finder(list)
        True

        ```
    """
    return _default_registry.get()


get_default_child_finder_registry = get_default_registry


def _register_default_child_finders(registry: ChildFinderRegistry) -> None:
    """Register default child finders for common Python types.

    This is the one place that says what is a container for the iterator package. The containers
    are the ones of ``coola.recursive``.

    Args:
        registry: The `ChildFinderRegistry` to populate with default child finders.

    Notes:
        This function is automatically called by `get_default_registry()` and should not
        be called directly by users.
    """
    default = DefaultChildFinder()
    iterable = IterableChildFinder()
    mapping = MappingChildFinder()
    # The finders are resolved with the MRO, so ``object`` alone would be enough. The scalars
    # are registered explicitly to document the types that must never be expanded, notably
    # ``str`` and ``bytes`` (they are ``Sequence``) and ``UserString`` (a ``Sequence`` of
    # ``UserString``: expanding it would never end).
    registry.register_many(
        {
            **dict.fromkeys((object, str, bytes, int, float, complex, bool, UserString), default),
            # Iterators and generators are not containers: they are leaves, so they are never
            # consumed (or iterated forever) by a traversal.
            **dict.fromkeys((list, tuple, deque, Sequence, set, frozenset), iterable),
            # ``MappingProxyType`` is a virtual member of ``Mapping`` (it does not inherit from
            # it), so it is registered explicitly.
            **dict.fromkeys((dict, Mapping, MappingProxyType), mapping),
        }
    )
    # Dataclasses without their own child finder
    registry.register_fallback(dataclasses.is_dataclass, DataclassChildFinder())


_default_registry: LazySingleton[ChildFinderRegistry] = make_default_registry_singleton(
    ChildFinderRegistry,
    _register_default_child_finders,
    plugin_group="coola.iterator.child_finders",
)
