r"""Define the child finder registry for managing iterators based on
data types."""

from __future__ import annotations

__all__ = ["ChildFinderRegistry"]

from collections import deque
from typing import TYPE_CHECKING, Any

from coola.iterator.bfs.default import DefaultChildFinder
from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from coola.iterator.bfs.base import BaseChildFinder


class ChildFinderRegistry:
    r"""Registry that manages iterators for different data types.

    This registry stores iterators for various data types and handles
    the dispatching of the appropriate iterator based on the data
    type during iteration. It uses Method Resolution Order (MRO) to
    resolve the most specific iterator for a given data type.
    It also supports caching of iterators for performance optimization
    in repetitive iteration tasks.

    Args:
        registry: An optional dictionary mapping types to iterators.
            If provided, the registry is initialized with this mapping.

    Attributes:
        _registry: Internal mapping of registered data types to iterators.
        _default_child_finder: The fallback iterator used for types not explicitly registered.
        _iterator_cache: Cache to speed up iterator lookups.

    Example:
        Basic usage:

        ```pycon
        >>> from coola.iterator.bfs import ChildFinderRegistry, IterableChildFinder
        >>> registry = ChildFinderRegistry({list: IterableChildFinder()})
        >>> registry
        ChildFinderRegistry(
          (<class 'list'>): IterableChildFinder()
        )
        >>> list(registry.iterate([1, 2, 3]))
        [1, 2, 3]

        ```

        Working with nested structures:

        ```pycon
        >>> from coola.iterator.bfs import get_default_registry
        >>> registry = get_default_registry()
        >>> data = {"a": [1, 2], "b": [3, 4]}
        >>> list(registry.iterate(data))
        [1, 2, 3, 4]

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
        r"""Register an iterator for a given data type.

        This method associates a specific iterator with a type.
        If data of this type is iterated, the registered iterator
        will be used. The cache is cleared after a registration to
        ensure consistency.

        Args:
            data_type: The Python type to register (e.g., `list`, `dict`, custom types).
            child_finder: The child finder instance that handles this type.
            exist_ok: If `True`, allows overwriting an existing registration.
                If `False`, raises an error.

        Raises:
            RuntimeError: If the type is already registered and `exist_ok` is `False`.

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

        This method allows for bulk registration of child finders for multiple data types.

        Args:
            mapping: A dictionary mapping Python types to their respective child finders.
            exist_ok: If `True`, allows overwriting existing registrations.

        Raises:
            RuntimeError: If any type is already registered and `exist_ok` is `False`.

        Example:
        ```pycon
        >>> from coola.iterator.bfs import (
        ...     ChildFinderRegistry,
        ...     IterableChildFinder,
        ...     MappingChildFinder,
        ... )
        >>> registry = ChildFinderRegistry()
        >>> registry.register_many({list: IterableChildFinder(), dict: MappingChildFinder()})
        >>> registry
        ChildFinderRegistry(
          (<class 'list'>): IterableChildFinder()
          (<class 'dict'>): MappingChildFinder()
        )

        ```
        """
        for typ, child_finder in mapping.items():
            self.register(typ, child_finder, exist_ok=exist_ok)

    def has_child_finder(self, data_type: type) -> bool:
        r"""Check if a child finder is registered for a given data type.

        This method checks for direct registration. Even if this method returns `False`,
        a suitable child finder might still be found using the MRO lookup.

        Args:
            data_type: The type to check.

        Returns:
            `True` if an child finder is registered for the type, `False` otherwise.

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
        r"""Find the child_finder for a data type without using cache.

        This method looks up the most specific child finder for the given type, starting
        with direct matches and then walking up the MRO to find an appropriate child finder.

        Args:
            data_type: The data type for which to find a child finder.

        Returns:
            The matching child finder instance for the type or a fallback child finder.
        """
        if data_type in self._registry:
            return self._registry[data_type]

        for base_type in data_type.__mro__:
            if base_type in self._registry:
                return self._registry[base_type]

        return self._default_child_finder

    def find_child_finder(self, data_type: type) -> BaseChildFinder[Any]:
        r"""Find the appropriate child finder for a given type.

        This method uses the MRO to find the most specific child finder. It caches the result
        for performance, so subsequent lookups are faster.

        Args:
            data_type: The data type for which to find a child finder.

        Returns:
            The appropriate child finder for the data type.

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
        r"""Perform breath-first iteration over a data structure.

        This method uses the appropriate child finder for the data type, which may be
        retrieved via the registry. The child finder will recursively traverse the data
        structure, yielding elements based on its specific implementation.

        Args:
            data: The data structure to iterate over.

        Yields:
            The elements of the data structure according to the
                appropriate child finder's traversal logic.

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
        >>> list(registry.iterate({"a": [1, 2], "b": [3, 4]}))
        [1, 2, 3, 4]

        ```
        """
        child_finder = self.find_child_finder(type(data))
        yield from child_finder.find_children(data)

    def iterate(self, data: Any) -> Iterator[Any]:
        r"""Perform depth-first iteration over a data structure.

        This method uses the appropriate child finder for the data type, which may be
        retrieved via the registry. The child finder will recursively traverse the data
        structure, yielding elements based on its specific implementation.

        Args:
            data: The data structure to iterate over.

        Yields:
            The elements of the data structure according to the
                appropriate child finder's traversal logic.

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
            children = list(self.find_children(current))
            if len(children) == 0:
                yield current
            queue.extend(children)
