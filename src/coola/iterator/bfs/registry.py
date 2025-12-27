r"""Define the child finder registry for managing child finders used for
breadth-first traversal of nested data structures."""

from __future__ import annotations

__all__ = ["ChildFinderRegistry"]

from collections import deque
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from coola.iterator.bfs.default import DefaultChildFinder
from coola.utils.format import repr_indent, repr_mapping, str_indent, str_mapping

if TYPE_CHECKING:
    from collections.abc import Iterator

    from coola.iterator.bfs.base import BaseChildFinder


class ChildFinderRegistry:
    r"""Registry that manages child finders for breadth-first traversal
    of nested data structures.

    This registry maps Python data types to ``BaseChildFinder`` instances.
    During traversal, the registry selects the most specific child finder
    for a given object using Method Resolution Order (MRO). If no match
    is found, a default child finder is used.

    The registry also caches resolved child finders to speed up repeated
    lookups.

    Args:
        registry: An optional dictionary mapping Python types to
            ``BaseChildFinder`` instances. If provided, the registry
            is initialized with this mapping.

    Attributes:
        _registry: Mapping of registered data types to child finders.
        _default_child_finder: Fallback child finder used when no match
            is found in the registry.
        _child_finder_cache: Cache mapping data types to resolved child
            finders (after MRO lookup).

    Example:
        Basic usage with a flat iterable:

        ```pycon
        >>> from coola.iterator.bfs import ChildFinderRegistry, IterableChildFinder
        >>> registry = ChildFinderRegistry({list: IterableChildFinder()})
        >>> list(registry.iterate([1, 2, 3]))
        [1, 2, 3]

        ```

        Working with nested structures using the default registry:

        ```pycon
        >>> from coola.iterator.bfs import get_default_registry
        >>> registry = get_default_registry()
        >>> data = {"a": [1, 2], "b": [3, 4]}
        >>> list(registry.iterate(data))
        [1, 2, 3, 4]

        ```

        Breadth-first traversal over mixed nested data:

        ```pycon
        >>> from coola.iterator.bfs import (
        ...     ChildFinderRegistry,
        ...     IterableChildFinder,
        ...     MappingChildFinder,
        ... )
        >>> registry = ChildFinderRegistry(
        ...     {list: IterableChildFinder(), dict: MappingChildFinder()}
        ... )
        >>> data = {"a": [1, 2], "b": [3, 4], "c": 5, "d": {"e": 6}}
        >>> list(registry.iterate(data))
        [5, 1, 2, 3, 4, 6]

        ```
    """

    def __init__(self, registry: dict[type, BaseChildFinder[Any]] | None = None) -> None:
        self._registry: dict[type, BaseChildFinder[Any]] = registry.copy() if registry else {}
        self._default_child_finder: BaseChildFinder[Any] = DefaultChildFinder()
        self._child_finder_cache: dict[type, BaseChildFinder[Any]] = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {repr_indent(repr_mapping(self._registry))}\n)"

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_mapping(self._registry))}\n)"

    def register(
        self,
        data_type: type,
        child_finder: BaseChildFinder[Any],
        exist_ok: bool = False,
    ) -> None:
        r"""Register a child finder for a given data type.

        This method associates a specific ``BaseChildFinder`` with a
        Python type. When an object of this type (or a subclass) is
        encountered during traversal, the registered child finder
        will be used.

        The internal cache is cleared after registration to ensure
        consistency.

        Args:
            data_type: The Python type to register (e.g., ``list``,
                ``dict``, or a custom class).
            child_finder: The child finder instance responsible for
                extracting children from objects of this type.
            exist_ok: If ``True``, allows overwriting an existing
                registration. If ``False``, raises an error.

        Raises:
            RuntimeError: If the type is already registered and
                ``exist_ok`` is ``False``.

        Example:
            ```pycon
            >>> from coola.iterator.bfs import ChildFinderRegistry, IterableChildFinder
            >>> registry = ChildFinderRegistry()
            >>> registry.register(list, IterableChildFinder())
            >>> registry.has_child_finder(list)
            True

            ```
        """
        if data_type in self._registry and not exist_ok:
            msg = (
                f"Child finder {self._registry[data_type]} already registered for {data_type}. "
                f"Use exist_ok=True to overwrite."
            )
            raise RuntimeError(msg)
        self._registry[data_type] = child_finder
        self._child_finder_cache.clear()

    def register_many(
        self,
        mapping: Mapping[type, BaseChildFinder[Any]],
        exist_ok: bool = False,
    ) -> None:
        r"""Register multiple child finders at once.

        Args:
            mapping: A mapping from Python types to their corresponding
                child finders.
            exist_ok: If ``True``, allows overwriting existing
                registrations.

        Raises:
            RuntimeError: If any type is already registered and
                ``exist_ok`` is ``False``.

        Example:
            ```pycon
            >>> from coola.iterator.bfs import (
            ...     ChildFinderRegistry,
            ...     IterableChildFinder,
            ...     MappingChildFinder,
            ... )
            >>> registry = ChildFinderRegistry()
            >>> registry.register_many({list: IterableChildFinder(), dict: MappingChildFinder()})
            >>> registry.has_child_finder(list), registry.has_child_finder(dict)
            (True, True)

            ```
        """
        for typ, child_finder in mapping.items():
            self.register(typ, child_finder, exist_ok=exist_ok)

    def has_child_finder(self, data_type: type) -> bool:
        r"""Check if a child finder is directly registered for a data
        type.

        This method only checks for an exact type match in the registry.
        Even if this returns ``False``, a suitable child finder may still
        be resolved via MRO lookup.

        Args:
            data_type: The type to check.

        Returns:
            ``True`` if a child finder is directly registered for the
                type, ``False`` otherwise.

        Example:
            ```pycon
            >>> from coola.iterator.bfs import ChildFinderRegistry, IterableChildFinder
            >>> registry = ChildFinderRegistry({list: IterableChildFinder()})
            >>> registry.has_child_finder(list)
            True
            >>> registry.has_child_finder(tuple)
            False

            ```
        """
        return data_type in self._registry

    def _find_child_finder_uncached(self, data_type: type) -> BaseChildFinder[Any]:
        r"""Resolve a child finder for a data type without using the
        cache.

        Resolution is performed by first checking for an exact match,
        then walking the type's Method Resolution Order (MRO) to find
        the most specific registered base class. If no match is found,
        the default child finder is returned.

        Args:
            data_type: The data type for which to resolve a child finder.

        Returns:
            A ``BaseChildFinder`` instance.
        """
        if data_type in self._registry:
            return self._registry[data_type]

        for base_type in data_type.__mro__:
            if base_type in self._registry:
                return self._registry[base_type]

        return self._default_child_finder

    def find_child_finder(self, data_type: type) -> BaseChildFinder[Any]:
        r"""Find the appropriate child finder for a given data type.

        This method resolves the child finder using MRO lookup and
        caches the result for faster subsequent access.

        Args:
            data_type: The data type for which to find a child finder.

        Returns:
            The resolved child finder instance.

        Example:
            ```pycon
            >>> from coola.iterator.bfs import ChildFinderRegistry, IterableChildFinder
            >>> registry = ChildFinderRegistry({list: IterableChildFinder()})
            >>> registry.find_child_finder(list)
            IterableChildFinder()
            >>> registry.find_child_finder(tuple)
            DefaultChildFinder()

            ```
        """
        if data_type not in self._child_finder_cache:
            self._child_finder_cache[data_type] = self._find_child_finder_uncached(data_type)
        return self._child_finder_cache[data_type]

    def find_children(self, data: Any) -> Iterator[Any]:
        r"""Return the immediate children of an object using its child
        finder.

        This method does not perform traversal by itself. It delegates
        to the appropriate child finder for the object's type.

        Args:
            data: The object whose children should be extracted.

        Yields:
            Child objects as defined by the resolved child finder.

        Example:
            ```pycon
            >>> from coola.iterator.bfs import ChildFinderRegistry, IterableChildFinder
            >>> registry = ChildFinderRegistry({list: IterableChildFinder()})
            >>> list(registry.find_children([1, 2, 3]))
            [1, 2, 3]

            ```
        """
        child_finder = self.find_child_finder(type(data))
        yield from child_finder.find_children(data)

    def iterate(self, data: Any) -> Iterator[Any]:
        r"""Perform a breadth-first traversal over a nested data
        structure.

        This method traverses the input data using breadth-first search
        (BFS). Container objects (mappings and iterables, excluding
        strings and bytes) are expanded using registered child finders.
        Only non-container (leaf) values are yielded.

        Containers themselves are never yielded, even if they are empty.

        Args:
            data: The data structure to traverse.

        Yields:
            Atomic (non-container) values in breadth-first order.

        Example:
            ```pycon
            >>> from coola.iterator.bfs import (
            ...     ChildFinderRegistry,
            ...     IterableChildFinder,
            ...     MappingChildFinder,
            ... )
            >>> registry = ChildFinderRegistry(
            ...     {list: IterableChildFinder(), dict: MappingChildFinder()}
            ... )
            >>> list(registry.iterate({"a": [1, 2], "b": [3, 4], "c": 5, "d": {"e": 6}}))
            [5, 1, 2, 3, 4, 6]

            ```
        """
        queue = deque([data])

        while queue:
            current = queue.popleft()
            is_container = isinstance(current, (Mapping, Iterable)) and not isinstance(
                current, (str, bytes)
            )
            children = list(self.find_children(current))
            if is_container or children:
                queue.extend(children)
            else:
                yield current
