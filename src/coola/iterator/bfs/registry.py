r"""Define the child finder registry for managing child finders used for
breadth-first traversal of nested data structures."""

from __future__ import annotations

__all__ = ["ChildFinderRegistry"]

from collections import deque
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from coola.iterator.bfs.base import BaseChildFinder
from coola.registry import TypeRegistry
from coola.utils.format import repr_indent, repr_mapping, str_indent, str_mapping

if TYPE_CHECKING:
    from collections.abc import Iterator


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
        initial_state: An optional dictionary mapping Python types to
            ``BaseChildFinder`` instances. If provided, the registry
            is initialized with this mapping.

    Attributes:
        _state: Mapping of registered data types to child finders.

    Example:
        Basic usage with a flat iterable:

        ```pycon
        >>> from coola.iterator.bfs import (
        ...     ChildFinderRegistry,
        ...     IterableChildFinder,
        ...     DefaultChildFinder,
        ... )
        >>> registry = ChildFinderRegistry(
        ...     {object: DefaultChildFinder(), list: IterableChildFinder()}
        ... )
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
        ...     {
        ...         object: DefaultChildFinder(),
        ...         list: IterableChildFinder(),
        ...         dict: MappingChildFinder(),
        ...     }
        ... )
        >>> data = {"a": [1, 2], "b": [3, 4], "c": 5, "d": {"e": 6}}
        >>> list(registry.iterate(data))
        [5, 1, 2, 3, 4, 6]

        ```
    """

    def __init__(self, initial_state: dict[type, BaseChildFinder[Any]] | None = None) -> None:
        self._state: TypeRegistry[BaseChildFinder] = TypeRegistry[BaseChildFinder](initial_state)

    def __repr__(self) -> str:
        state = repr_indent(repr_mapping({"state": self._state}))
        return f"{self.__class__.__qualname__}(\n  {state}\n)"

    def __str__(self) -> str:
        state = str_indent(str_mapping({"state": self._state}))
        return f"{self.__class__.__qualname__}(\n  {state}\n)"

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
        self._state.register(data_type, child_finder, exist_ok)

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
        self._state.register_many(mapping, exist_ok)

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
        return data_type in self._state

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
            >>> from coola.iterator.bfs import (
            ...     ChildFinderRegistry,
            ...     IterableChildFinder,
            ...     DefaultChildFinder,
            ... )
            >>> registry = ChildFinderRegistry(
            ...     {object: DefaultChildFinder(), list: IterableChildFinder()}
            ... )
            >>> registry.find_child_finder(list)
            IterableChildFinder()
            >>> registry.find_child_finder(tuple)
            DefaultChildFinder()

            ```
        """
        return self._state.resolve(data_type)

    def find_children(self, data: object) -> Iterator[Any]:
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
            >>> from coola.iterator.bfs import (
            ...     ChildFinderRegistry,
            ...     IterableChildFinder,
            ...     DefaultChildFinder,
            ... )
            >>> registry = ChildFinderRegistry(
            ...     {object: DefaultChildFinder(), list: IterableChildFinder()}
            ... )
            >>> list(registry.find_children([1, 2, 3]))
            [1, 2, 3]

            ```
        """
        child_finder = self.find_child_finder(type(data))
        yield from child_finder.find_children(data)

    def iterate(self, data: object) -> Iterator[Any]:
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
            ...     DefaultChildFinder,
            ... )
            >>> registry = ChildFinderRegistry(
            ...     {
            ...         object: DefaultChildFinder(),
            ...         list: IterableChildFinder(),
            ...         dict: MappingChildFinder(),
            ...     }
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
